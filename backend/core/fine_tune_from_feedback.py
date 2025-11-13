# fine_tune_from_feedback.py
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import json
from PIL import Image

# Simple dataset loader for feedback images (frame_path -> image)
class FeedbackDataset(Dataset):
    def __init__(self, entries, label2idx, transform=None):
        self.entries = entries
        self.label2idx = label2idx
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e['frame_path']).convert('RGB')
        x = self.transform(img)
        y = self.label2idx[e['feedback_label']]
        return x, y

def load_feedback_entries(db_conn, min_count=1):
    # Pseudo: query feedbacks joined with detections where feedback_type in ('confirm','correct')
    # Return list of dicts: {frame_path, feedback_label}
    # Implement based on your DB
    pass

def fine_tune_model(entries, label2idx, base_model_path, out_path, epochs=3, batch=16, lr=1e-4):
    ds = FeedbackDataset(entries, label2idx)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    # load pre-trained model
    model = models.resnet18(pretrained=False)
    state = torch.load(base_model_path)
    model.load_state_dict(state)
    # replace classifier
    num_classes = len(label2idx)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss {total_loss/len(loader)}")
    # save
    torch.save(model.state_dict(), out_path)
    print("Saved fine tuned model:", out_path)
