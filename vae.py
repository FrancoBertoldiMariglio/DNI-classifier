import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

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


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAEEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 224x224 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 112x112 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 56x56 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 7, 7)
        x = self.deconv(x)
        return x


class DNIVAEDetector:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = VAEEncoder().to(self.device)
        self.decoder = VAEDecoder().to(self.device)
        self.thresholds = {}
        self.training = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.mse_criterion = nn.MSELoss()
        self.ssim_criterion = SSIM().to(self.device)
        self.perceptual_criterion = PerceptualLoss().to(self.device)

    def reparameterize(self, mu, logvar):
        if self.encoder.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def compute_loss(self, reconstructed, original, mu, logvar, loss_weights):
        """
        Calcula la pérdida total del VAE
        """
        # Pérdida de reconstrucción
        mse_loss = self.mse_criterion(reconstructed, original)
        ssim_loss = 1 - self.ssim_criterion(reconstructed, original)
        perceptual_loss = self.perceptual_criterion(reconstructed, original)

        # Pérdida de reconstrucción ponderada
        recon_loss = (loss_weights["mse"] * mse_loss +
                      loss_weights["ssim"] * ssim_loss +
                      loss_weights["perceptual"] * perceptual_loss)

        # KL Divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Pérdida total (con factor beta para KLD)
        total_loss = recon_loss + loss_weights.get("kld", 0.1) * kld_loss

        return total_loss, recon_loss, kld_loss

    def train_model(self, data_dir, model_name, epochs=30, batch_size=32, learning_rate=1e-3, loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                'mse': 0.7,
                'ssim': 0.2,
                'perceptual': 0.1,
                'kld': 0.1  # Factor beta para KL Divergence
            }

        dataset = DNIDataset(data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        patience = 5
        no_improve_count = 0

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Dataset cargado: {len(train_dataset)} imágenes de entrenamiento, {len(val_dataset)} de validación")

        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                lr=learning_rate,
                                weight_decay=1e-4,
                                betas=(0.9, 0.999))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch + 1}/{epochs}]")

            # Training
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            batch_losses = []
            recon_losses = []
            kld_losses = []

            train_bar = tqdm(train_loader, desc="Training")
            for batch in train_bar:
                imgs = batch.to(self.device)

                # Forward pass
                mu, logvar = self.encoder(imgs)
                z = self.reparameterize(mu, logvar)
                reconstructed = self.decoder(z)

                # Calcular pérdidas
                loss, recon_loss, kld_loss = self.compute_loss(
                    reconstructed, imgs, mu, logvar, loss_weights
                )

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Guardar métricas
                batch_losses.append(loss.item())
                recon_losses.append(recon_loss.item())
                kld_losses.append(kld_loss.item())
                epoch_loss += loss.item()

                train_bar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'avg_loss': f'{np.mean(batch_losses):.4f}'
                })

            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation
            self.encoder.eval()
            self.decoder.eval()
            val_loss = 0
            val_recon_losses = []
            val_kld_losses = []
            reconstruction_errors = []

            val_bar = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for batch in val_bar:
                    imgs = batch.to(self.device)

                    mu, logvar = self.encoder(imgs)
                    z = self.reparameterize(mu, logvar)
                    reconstructed = self.decoder(z)

                    loss, recon_loss, kld_loss = self.compute_loss(
                        reconstructed, imgs, mu, logvar, loss_weights
                    )

                    val_recon_losses.append(recon_loss.item())
                    val_kld_losses.append(kld_loss.item())
                    reconstruction_errors.append(loss.item())
                    val_loss += loss.item()

                    val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            # Calcular umbral de anomalía
            self.thresholds = {
                '90': np.percentile(reconstruction_errors, 90),
                '95': np.percentile(reconstruction_errors, 95),
                '99': np.percentile(reconstruction_errors, 99)
            }

            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

            # Actualizar learning rate
            scheduler.step()

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model_name)
                print(f"✓ Nuevo mejor modelo guardado (val_loss: {val_loss:.6f})")
                no_improve_count = 0
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
        print(f"Umbrales de anomalía:")
        for percentile, value in self.thresholds.items():
            print(f"  {percentile}%: {value:.6f}")

        return train_losses, val_losses

    def predict(self, image_path, threshold_option, loss_weights=None, return_images=False):
        if loss_weights is None:
            loss_weights = {
                'mse': 0.6,
                'ssim': 0.25,
                'perceptual': 0.15,
                'kld': 0.1
            }

        if self.thresholds is None:
            raise ValueError("Model needs to be trained first to establish threshold")

        valid_thresholds = ['90', '95', '99']
        if threshold_option not in valid_thresholds:
            raise ValueError(f"Threshold option must be one of {valid_thresholds}")

        threshold = self.thresholds[threshold_option]

        self.encoder.eval()
        self.decoder.eval()

        # Prepare image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, logvar = self.encoder(image_tensor)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decoder(z)

            loss, recon_loss, kld_loss = self.compute_loss(
                reconstructed, image_tensor, mu, logvar, loss_weights
            )

            reconstruction_error = loss.item()

            # Calcular score de confianza
            normalized_error = reconstruction_error / threshold
            confidence = 1 / (1 + np.exp(5 * (normalized_error - 1)))

        if return_images:
            return confidence, reconstruction_error, image_tensor.squeeze(0), reconstructed.squeeze(0)
        return confidence, reconstruction_error

    def save_model(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'thresholds': self.thresholds
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.thresholds = checkpoint['thresholds']


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


class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv3_2', 'conv4_2'], requires_grad=False):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad = False

        self.layers = layers
        self.selected_layers = {
            'conv1_2': 3,
            'conv2_2': 8,
            'conv3_2': 15,
            'conv4_2': 22,
            'conv5_2': 29
        }
        self.model = nn.Sequential(*[vgg[i] for i in range(max(self.selected_layers[layer] for layer in layers) + 1)])

    def forward(self, x, y):
        features_x = {}
        features_y = {}

        for name, layer in zip(self.selected_layers, self.model):
            x = layer(x)
            y = layer(y)

            if name in self.layers:
                features_x[name] = x
                features_y[name] = y

        loss = 0
        for layer in self.layers:
            loss += torch.nn.functional.mse_loss(features_x[layer], features_y[layer])

        return loss


def main():
    # Inicializar el detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DNIVAEDetector(device=device)

    # Configurar pesos de pérdida
    loss_weights = {
        'mse': 0.7,
        'ssim': 0.2,
        'perceptual': 0.1,
        'kld': 0.1
    }

    # Entrenar el modelo
    train_losses, val_losses = detector.train_model(
        data_dir="autoencoder_data/train_cropped",
        model_name="vae_model.pt",
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        loss_weights=loss_weights
    )

    # Ejemplo de predicción
    confidence, error = detector.predict(
        image_path="test/original/valid/0a1e89f6-0ae2-4cab-854e-61c897cbbe13.jpg",
        threshold_option='95',
        loss_weights=loss_weights
    )

    print(f"Confidence: {confidence:.4f}")
    print(f"Reconstruction Error: {error:.4f}")


if __name__ == "__main__":
    main()