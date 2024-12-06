import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributed.checkpoint import load_state_dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from math import exp
import torch.nn.functional as torch_functional
from tqdm import tqdm


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
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 28 * 28, 256)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 128 * 28 * 28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 28, 28)
        x = self.deconv(x)
        return x


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
        ])

    def train_model(self, data_dir, epochs=30, batch_size=64, learning_rate=1e-3, loss_weights=None):
        if loss_weights is None:
            loss_weights = {'mse': 1.0, 'ssim': 0.0}

        # Inicializar wandb con nuevos hiperparámetros
        run = wandb.init(
            project="DNI Anomaly Detector - Autoencoder",
            config={
                "architecture": "Lightweight Autoencoder",
                "dataset": data_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": 1e-4,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "loss_weights": {
                    "mse": loss_weights["mse"],
                    "ssim": loss_weights["ssim"]
                },
                "latent_dim": 256,
                "conv_channels": [32, 64, 128]
            },
            name="DNI Anomaly Detector - Lightweight"
        )

        dataset = DNIDataset(data_dir, self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        patience = 5
        no_improve_count = 0

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Dataset cargado: {len(train_dataset)} imágenes de entrenamiento, {len(val_dataset)} de validación")
        wandb.log({"train_size": len(train_dataset), "val_size": len(val_dataset)})

        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        mse_criterion = nn.MSELoss()
        ssim_criterion = SSIM()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            epoch_metrics = {'epoch': epoch + 1}

            # Training
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            batch_losses = []
            mse_losses = []
            ssim_losses = []

            train_bar = tqdm(train_loader, desc="Training")
            for batch in train_bar:
                imgs = batch.to(self.device)

                latent = self.encoder(imgs)
                reconstructed = self.decoder(latent)

                mse_loss = mse_criterion(reconstructed, imgs)
                ssim_loss = 1 - ssim_criterion(reconstructed, imgs)
                loss = loss_weights["mse"] * mse_loss + loss_weights["ssim"] * ssim_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                mse_losses.append(mse_loss.item())
                ssim_losses.append(ssim_loss.item())
                epoch_loss += loss.item()

                train_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{np.mean(batch_losses):.4f}'
                })

            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            # Logging métricas de entrenamiento
            epoch_metrics.update({
                'train_loss': train_loss,
                'train_mse': np.mean(mse_losses),
                'train_ssim': np.mean(ssim_losses),
            })

            # Validation
            self.encoder.eval()
            self.decoder.eval()
            val_loss = 0
            val_mse_losses = []
            val_ssim_losses = []
            reconstruction_errors = []

            # Validation
            val_bar = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for batch in val_bar:
                    imgs = batch.to(self.device)
                    latent = self.encoder(imgs)
                    reconstructed = self.decoder(latent)

                    # Calcular pérdidas
                    mse_loss = mse_criterion(reconstructed, imgs)
                    ssim_loss = 1 - ssim_criterion(reconstructed, imgs)
                    loss = loss_weights["mse"] * mse_loss + loss_weights["ssim"] * ssim_loss

                    val_mse_losses.append(mse_loss.item())
                    val_ssim_losses.append(ssim_loss.item())
                    reconstruction_errors.append(mse_loss.item())
                    val_loss += loss.item()

                    val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            # Calcular umbral de anomalía
            self.threshold = np.percentile(reconstruction_errors, 95)
            wandb.run.summary["anomaly_threshold"] = self.threshold

            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

            # Logging métricas de validación
            epoch_metrics.update({
                'val_loss': val_loss,
                'val_mse': np.mean(val_mse_losses),
                'val_ssim': np.mean(val_ssim_losses),
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Log métricas a wandb
            wandb.log(epoch_metrics)

            # Actualizar learning rate
            scheduler.step(val_loss)

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('dni_anomaly_detector.pt')
                print(f"✓ Nuevo mejor modelo guardado (val_loss: {val_loss:.6f})")
                wandb.run.summary["best_val_loss"] = val_loss
                # Guardar mejor modelo en wandb
                artifact = wandb.Artifact(
                    f"best_model_{wandb.run.id}", type="model",
                    description=f"Mejor modelo con val_loss: {val_loss:.6f}"
                )
                artifact.add_file('dni_anomaly_detector_old.pt')
                wandb.log_artifact(artifact)
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"✗ No hubo mejora en {patience} epochs, deteniendo entrenamiento")
                break

            print(f"Epoch {epoch + 1} completada:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        print("\nEntrenamiento completado:")
        print(f"Mejor val_loss: {best_val_loss:.6f}")
        print(f"Umbral de anomalía: {self.threshold:.6f}")

        # Cerrar wandb
        wandb.finish()

        return train_losses, val_losses

    def predict(self, image_path, loss_weights=None):
        if loss_weights is None:
            loss_weights = {'mse': 1.0, 'ssim': 0.0}
        if self.threshold is None:
            raise ValueError("Model needs to be trained first to establish threshold")

        self.encoder.eval()
        self.decoder.eval()

        # Prepare image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.encoder(image_tensor)
            reconstructed = self.decoder(latent)

            # Usar MSELoss igual que en entrenamiento
            mse_criterion = nn.MSELoss()
            reconstruction_error = mse_criterion(reconstructed, image_tensor).item()

            # Calcular score de confianza usando sigmoide
            normalized_error = reconstruction_error / self.threshold
            confidence = 1 / (1 + np.exp(5 * (normalized_error - 1)))

        return confidence, reconstruction_error

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

    # Entrenar el modelo
    train_losses, val_losses = detector.train_model(
        data_dir="recortes",
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        loss_weights={'mse': 0.8, 'ssim': 0.2}
    )

if __name__ == "__main__":
    main()