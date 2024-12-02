import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.checkpoint import load_state_dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from math import exp
import torch.nn.functional as torch_functional


class DNIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('*.jpg'))
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Primer bloque
            nn.Conv2d(3, 48, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Segundo bloque
            nn.Conv2d(48, 48, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Tercer bloque
            nn.Conv2d(48, 48, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Cuarto bloque
            nn.Conv2d(48, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Quinto bloque
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # Primer bloque transpuesto
            nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Segundo bloque transpuesto
            nn.ConvTranspose2d(48, 48, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Tercer bloque transpuesto
            nn.ConvTranspose2d(48, 48, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Cuarto bloque transpuesto
            nn.ConvTranspose2d(48, 48, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Capa de salida
            nn.Conv2d(48, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class DNIAnomalyDetector:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.threshold = None

        # Mejores transformaciones para DNIs
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def train_model(self, data_dir, epochs=50, batch_size=32, learning_rate=1e-3):
        dataset = DNIDataset(data_dir, self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizador mejorado con scheduling
        optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Pérdida combinada: MSE + SSIM
        mse_criterion = nn.MSELoss(reduction='none')
        ssim_criterion = SSIM()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0

            for batch in train_loader:
                imgs = batch.to(self.device)

                # Forward pass
                latent = self.encoder(imgs)
                reconstructed = self.decoder(latent)

                # Calcular pérdidas
                mse_loss = mse_criterion(reconstructed, imgs).mean()
                ssim_loss = 1 - ssim_criterion(reconstructed, imgs)
                loss = 0.7 * mse_loss + 0.3 * ssim_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # Validation
            self.encoder.eval()
            self.decoder.eval()
            val_loss = 0
            reconstruction_errors = []

            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch.to(self.device)
                    latent = self.encoder(imgs)
                    reconstructed = self.decoder(latent)

                    mse_loss = mse_criterion(reconstructed, imgs).mean(dim=(1, 2, 3))
                    ssim_loss = 1 - ssim_criterion(reconstructed, imgs)
                    loss = 0.7 * mse_loss.mean() + 0.3 * ssim_loss

                    reconstruction_errors.extend(mse_loss.cpu().numpy())
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

            # Actualizar learning rate
            scheduler.step(val_loss)

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pt')

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {train_losses[-1]:.6f}, '
                      f'Val Loss: {val_loss:.6f}')

        # Calcular umbral de anomalía
        self.threshold = np.percentile(reconstruction_errors, 95)

        return train_losses, val_losses

    def predict(self, image_path):
        self.encoder.eval()
        self.decoder.eval()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.encoder(image)
            reconstructed = self.decoder(latent)

            mse_loss = nn.MSELoss(reduction='mean')(reconstructed, image).item()
            confidence = 1 - min(mse_loss / self.threshold, 1)

        return confidence

    def save_model(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'threshold': self.threshold
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.threshold = checkpoint['threshold']


# SSIM Loss
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch_functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = torch_functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch_functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch_functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch_functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


# Ejemplo de uso
def main():
    # Inicializar el detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DNIAnomalyDetector(device=device)

    # # Entrenar el modelo
    # train_losses, val_losses = detector.train_model(
    #     data_dir="valid",
    #     epochs=50,
    #     batch_size=32
    # )
    #
    # # Guardar el modelo entrenado
    # detector.save_model("dni_anomaly_detector.pth")

    # Cargar el modelo entrenado
    detector.load_model("dni_anomaly_detector.pth")

    # Hacer una predicción
    confidence_invalid = detector.predict("IMG_20241023_122633215.jpg")
    print(f"Confidence_invalid score: {confidence_invalid:.2f}")


    # Hacer una predicción
    confidence_valid = detector.predict("0a1e89f6-0ae2-4cab-854e-61c897cbbe13.jpg")
    print(f"Confidence_valid score: {confidence_valid:.2f}")

if __name__ == "__main__":
    main()