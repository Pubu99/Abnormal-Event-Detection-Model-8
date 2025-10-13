"""
Sharpness-Aware Minimization (SAM) Optimizer
Seeks parameters in neighborhoods with uniformly low loss for better generalization.

Reference: Sharpness-Aware Minimization for Efficiently Improving Generalization
https://arxiv.org/abs/2010.01412
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    
    SAM simultaneously minimizes loss value and loss sharpness.
    It seeks parameters that lie in neighborhoods having uniformly low loss.
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
            rho: Neighborhood size for sharpness
            adaptive: Use adaptive SAM (ASAM)
            **kwargs: Arguments passed to base optimizer
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: ascend to find adversarial parameters.
        Should be called after loss.backward() but before optimizer.step().
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # Store old parameters
                self.state[p]["old_p"] = p.data.clone()
                
                # Compute and store epsilon = rho * grad / ||grad||
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: descend from adversarial parameters.
        Should be called after second loss.backward().
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # Get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]
        
        # Do the actual "sharpness-aware" update
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Single optimization step (combines both steps).
        Requires closure that reevaluates the model and returns the loss.
        """
        assert closure is not None, "SAM requires closure, but it was not provided"
        
        # Enable gradient computation for closure
        closure = torch.enable_grad()(closure)
        
        # First forward-backward pass
        loss = closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        self.second_step()
        
        return loss
    
    def _grad_norm(self):
        """Compute the norm of gradients."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class ASAM(SAM):
    """
    Adaptive Sharpness-Aware Minimization (ASAM).
    Variant of SAM that adapts to parameter scale.
    
    Reference: ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning
    https://arxiv.org/abs/2102.11600
    """
    
    def __init__(self, params, base_optimizer, rho=0.5, **kwargs):
        """
        Initialize ASAM optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class
            rho: Neighborhood size (typically larger than SAM)
            **kwargs: Arguments passed to base optimizer
        """
        super(ASAM, self).__init__(params, base_optimizer, rho=rho, adaptive=True, **kwargs)


def create_sam_optimizer(model, config):
    """
    Create SAM optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration object
        
    Returns:
        SAM optimizer instance
    """
    # Base optimizer parameters
    opt_config = config.training.optimizer
    
    # Choose base optimizer
    if opt_config.type == 'adamw':
        from torch.optim import AdamW
        base_optimizer = AdamW
    elif opt_config.type == 'sgd':
        from torch.optim import SGD
        base_optimizer = SGD
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config.type}")
    
    # SAM-specific parameters
    sam_config = config.training.get('sam', {})
    rho = sam_config.get('rho', 0.05)
    adaptive = sam_config.get('adaptive', False)
    
    # Create SAM optimizer
    if adaptive:
        optimizer = ASAM(
            model.parameters(),
            base_optimizer,
            rho=rho,
            lr=config.training.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)) if opt_config.type == 'adamw' else None
        )
    else:
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=rho,
            lr=config.training.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)) if opt_config.type == 'adamw' else None
        )
    
    return optimizer
