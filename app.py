import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tools.models import ResNet18
from tools.utils import load_images
from tools.train_val_test import run
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.optim as optim

app = FastAPI()


# Define the request body model for prediction
class ImageRequest(BaseModel):
    image_base64: str


# Define the same transformations used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to input size of ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])


# Endpoint for prediction
@app.post("/predict")
async def predict(request: ImageRequest):

    # Load the model for prediction endpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18()

    try:
        model.load_state_dict(torch.load('best_modelResNet18_dni_adjusted.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not found")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Apply transformations and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform the prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()

        return {"probability": probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for training
@app.post("/train")
async def train_model():
    try:
        # Configurations
        valid_images_path = 'DNI-Validos'
        invalid_images_path = 'DNI-Invalidos'
        test_size = 0.2
        val_size = 0.2
        seed = 42
        input_size = (224, 224)
        batch_size = 64
        num_epochs = 10
        learning_rate = 0.001
        model_name = 'ResNet18_dni_adjusted'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load image datasets with labels
        valid_images_dataset = load_images(valid_images_path, '1')
        invalid_images_dataset = load_images(invalid_images_path, '0')
        mydataset = pd.concat([valid_images_dataset, invalid_images_dataset], ignore_index=True)

        # Split datasets
        train_val_df, test_df = train_test_split(mydataset, test_size=test_size, stratify=mydataset['etiqueta'], random_state=seed)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['etiqueta'], random_state=seed)

        # Define the Dataset class
        class DniDataset(Dataset):
            def __init__(self, df, transform=None):
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                img_path = self.df.iloc[idx]['path']
                label = float(self.df.iloc[idx]['etiqueta'])
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                label = torch.tensor(label, dtype=torch.float32)
                return image, label

        # Create datasets and dataloaders
        train_dataset = DniDataset(train_df, transform=transform)
        val_dataset = DniDataset(val_df, transform=transform)
        test_dataset = DniDataset(test_df, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

        # Initialize the model, criterion, and optimizer
        train_model = ResNet18().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(train_model.parameters(), lr=learning_rate)

        # Run training
        run(train_dataloader, val_dataloader, test_dataloader, train_model, criterion, optimizer, model_name, device, num_epochs)

        return {"message": "Training completed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
