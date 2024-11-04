import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Definición del Autoencoder
class DNIAutoencoder(nn.Module):
    def __init__(self):
        super(DNIAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Dataset personalizado
class DNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# Función de entrenamiento
def train_autoencoder(model, train_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# Función para calcular el error de reconstrucción
def compute_reconstruction_error(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        reconstructed = model(image_tensor)
        error = torch.mean((image_tensor - reconstructed) ** 2).item()
    return error

# Función para cargar el modelo y el umbral
def load_model_and_threshold(model_path, device):
    model = DNIAutoencoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']
    return model, threshold

# Clasificación de una nueva imagen
def classify_dni_image(model, image_path, threshold, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    error = compute_reconstruction_error(model, image_tensor, device)
    is_valid = error <= threshold
    
    return is_valid, error

# Función principal para entrenar el modelo
def main():
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224  # Ajusta según el tamaño de tus imágenes
    batch_size = 32
    num_epochs = 50
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    # Crear dataset y dataloader
    dataset = DNIDataset(
        root_dir='ruta/a/tus/imagenes/validas',
        transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Crear y entrenar el modelo
    model = DNIAutoencoder().to(device)
    train_autoencoder(model, train_loader, num_epochs, device)
    
    # Calcular umbral de error usando el conjunto de entrenamiento
    errors = []
    for data in train_loader:
        data = data.to(device)
        for img in data:
            error = compute_reconstruction_error(model, img, device)
            errors.append(error)
    
    # Establecer umbral (por ejemplo, media + 2 desviaciones estándar)
    threshold = np.mean(errors) + 2 * np.std(errors)
    
    # Guardar el modelo y el umbral
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold
    }, 'dni_autoencoder.pth')

# Ejecución de la clasificación
if __name__ == '__main__':
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo y umbral
    model, threshold = load_model_and_threshold('dni_autoencoder.pth', device)
    
    # Ruta de la imagen a clasificar
    image_path = 'ruta/a/tu/imagen.jpg'  # Cambia esta ruta por la imagen que deseas probar
    
    # Clasificar imagen
    is_valid, error = classify_dni_image(model, image_path, threshold, device)
    
    if is_valid:
        print(f'La imagen es válida con un error de reconstrucción de {error:.6f}.')
    else:
        print(f'La imagen es inválida con un error de reconstrucción de {error:.6f}.')
