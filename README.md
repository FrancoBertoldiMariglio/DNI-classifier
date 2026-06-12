# DNI-classifier

End-to-end pipeline for classifying Argentine ID documents (DNI) as **valid or invalid**: a fine-tuned **ResNet18** (PyTorch) served through a **FastAPI** endpoint, packaged with **Docker**.

## How it works

**Training** (`create_model.py`)
- Loads valid/invalid document images and builds a stratified train / val / test split (60/20/20, fixed seed)
- Fine-tunes ResNet18 (224×224 input, ImageNet normalization)
- Hyperparameters centralized in `config.json` (batch size 64, lr 1e-3, 10 epochs)
- Saves the best checkpoint by validation performance

**Inference** (`app.py`)
- `POST /predict` receives a base64-encoded image and returns the predicted class
- Loads the trained checkpoint, applies the same transforms as training
- Runs on GPU when available, falls back to CPU

## Run it

```bash
# Train
python create_model.py

# Serve
uvicorn app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t dni-classifier .
docker run -p 8000:8000 dni-classifier
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-image>"}'
```

> Note: the document dataset and trained weights are private and not included.
