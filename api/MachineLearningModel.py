import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


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

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.threshold = checkpoint['threshold']
        print("THRESHOLD", self.threshold)
