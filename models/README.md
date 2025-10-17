# Models Directory

## 📦 Trained Model Checkpoints

Place your trained `.pth` model files here.

### Current Models:

- `best_model.pth` - Best performing model (99.38% accuracy)
- `last_model.pth` - Latest training checkpoint

### Model Information:

- Architecture: EfficientNet-B0 + BiLSTM + Transformer
- Input: 16-frame sequences (224x224 RGB)
- Output: 14 classes (13 anomalies + Normal)
- Parameters: ~15M
- Accuracy: 99.38% on UCF Crime test set

### Usage:

```python
from inference.engine import AnomalyDetector

detector = AnomalyDetector(model_path='models/best_model.pth')
result = detector.predict_video('path/to/video.mp4')
```

**NOTE:** Move your `outputs/checkpoints/best.pth` file to `models/best_model.pth`
