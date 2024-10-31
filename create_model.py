import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class IDOneClassClassifier(nn.Module):
    def __init__(self, pretrained=True, feature_extract=True):
        """
        One-Class Classification model for ID images using ResNet

        Args:
            pretrained (bool): Use pretrained weights
            feature_extract (bool): Only train final layers
        """
        super(IDOneClassClassifier, self).__init__()

        # Use ResNet-50 as base model
        self.base_model = models.resnet50(pretrained=pretrained)

        # Freeze base model parameters if feature extracting
        if feature_extract:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Modify the final layers
        num_ftrs = self.base_model.fc.in_features

        # Custom feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Remove original fully connected layer
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Extracted features
        """
        # Extract base features
        base_features = self.base_model(x)

        # Apply custom feature extraction
        features = self.feature_extractor(base_features)

        return features


class IDOneClassTrainer:
    def __init__(self, input_shape=(3, 224, 224), nu=0.1,
                 device='mps' if torch.mps.is_available() else 'cpu'):
        """
        Initialize One-Class Classification Trainer for ID images

        Args:
            input_shape (tuple): Input image shape
            nu (float): Anomaly sensitivity parameter
            device (str): Computing device
        """
        self.device = torch.device(device)
        self.model = IDOneClassClassifier().to(self.device)
        self.nu = nu
        self.radius = torch.tensor(0.).to(self.device)
        self.center = None

        # Data augmentation for ID images
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.augmentations.transforms.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=15, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _preprocess_data(self, X):
        """
        Preprocess input images

        Args:
            X (numpy.ndarray): Input images

        Returns:
            torch.Tensor: Preprocessed and augmented images
        """
        # Apply transformations to batch of images
        transformed_images = []
        for img in X:
            # Ensure the image is in the right format (HWC)
            if img.shape[0] in [1, 3]:  # If CHW, transpose
                img = img.transpose(1, 2, 0)

            # Convert to uint8 if not already
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)

            # Apply augmentation
            transformed = self.transform(image=img)
            transformed_images.append(transformed['image'])

        return torch.stack(transformed_images).to(self.device)

    def train(self, X, epochs=100, lr=1e-4, weight_decay=1e-5):
        """
        Train the one-class classification model

        Args:
            X (numpy.ndarray): Training data (normal ID images)
            epochs (int): Number of training epochs
            lr (float): Learning rate
            weight_decay (float): L2 regularization strength
        """
        # Prepare data
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        # Prepare optimizer
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            # Preprocess training data
            X_train_processed = self._preprocess_data(X_train)

            optimizer.zero_grad()

            # Extract features (remove `torch.no_grad()` here)
            outputs = self.model(X_train_processed)

            # Compute center of the hypersphere
            center = torch.mean(outputs, dim=0)

            # Compute distances
            distances = torch.sum((outputs - center) ** 2, dim=1)

            # Compute radius
            sorted_distances = torch.sort(distances)[0]
            radius = sorted_distances[int(self.nu * len(sorted_distances))]

            # Compute loss
            loss = torch.mean(torch.clamp(distances - radius ** 2, min=0))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                X_val_processed = self._preprocess_data(X_val)

                with torch.no_grad():
                    val_outputs = self.model(X_val_processed)
                    val_distances = torch.sum((val_outputs - center) ** 2, dim=1)

                print(f'Epoch [{epoch}/{epochs}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Radius: {radius.item():.4f}')

        # Store center and radius
        self.center = center.detach()
        self.radius = radius.detach()

    def predict(self, X, threshold_multiplier=1.0):
        """
        Predict if samples are within the learned hypersphere

        Args:
            X (numpy.ndarray): Input ID images
            threshold_multiplier (float): Adjust anomaly sensitivity

        Returns:
            numpy.ndarray: Binary predictions (1 for normal, 0 for anomaly)
        """
        self.model.eval()

        # Preprocess input data
        X_processed = self._preprocess_data(X)

        with torch.no_grad():
            # Extract features
            outputs = self.model(X_processed)

            # Compute distances
            distances = torch.sum((outputs - self.center) ** 2, dim=1)

            # Apply threshold with multiplier
            predictions = (distances <= (self.radius * threshold_multiplier) ** 2)

        return predictions.cpu().numpy().astype(int)

    def extract_features(self, X):
        """
        Extract features for further analysis

        Args:
            X (numpy.ndarray): Input ID images

        Returns:
            numpy.ndarray: Extracted features
        """
        self.model.eval()

        # Preprocess input data
        X_processed = self._preprocess_data(X)

        with torch.no_grad():
            features = self.model(X_processed)

        return features.cpu().numpy()


def load_images_from_directory(directory, max_images=None):
    """
    Load images from a specified directory

    Args:
        directory (str): Path to directory containing images
        max_images (int, optional): Maximum number of images to load

    Returns:
        numpy.ndarray: Array of loaded images
    """
    # Image loading transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    images = []
    # Support common image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # Iterate through files in directory
    for filename in os.listdir(directory):
        # Check if file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                # Open and transform image
                img_path = os.path.join(directory, filename)
                with Image.open(img_path) as img:
                    # Convert to RGB to handle different image modes
                    img = img.convert('RGB')
                    # Transform and convert to numpy
                    tensor_img = transform(img)
                    images.append(tensor_img.numpy())

                # Stop if max images is reached
                if max_images and len(images) >= max_images:
                    break

            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    # Convert to numpy array
    return np.array(images)


def example_usage(dataset_path='dataset',
                  valid_dir='valid',
                  invalid_dir='invalid',
                  max_valid_images=100,
                  max_invalid_images=20):
    """
    Train and test one-class classifier using images from directories

    Args:
        dataset_path (str): Root directory containing image subdirectories
        valid_dir (str): Directory name for valid (normal) images
        invalid_dir (str): Directory name for invalid (anomaly) images
        max_valid_images (int): Maximum number of valid images to load
        max_invalid_images (int): Maximum number of invalid images to load
    """
    # Construct full paths
    valid_path = os.path.join(dataset_path, valid_dir)
    invalid_path = os.path.join(dataset_path, invalid_dir)

    # Validate directories exist
    if not os.path.exists(valid_path):
        raise ValueError(f"Valid images directory not found: {valid_path}")
    if not os.path.exists(invalid_path):
        raise ValueError(f"Invalid images directory not found: {invalid_path}")

    # Load images
    print("Loading valid images...")
    normal_ids = load_images_from_directory(valid_path, max_valid_images)

    print("Loading invalid images...")
    anomalous_ids = load_images_from_directory(invalid_path, max_invalid_images)

    # Validate image loading
    if len(normal_ids) == 0:
        raise ValueError("No valid images found in the directory")

    # Initialize one-class classifier
    trainer = IDOneClassTrainer(input_shape=(3, 224, 224), nu=0.1)

    print(f"Training on {len(normal_ids)} normal ID images...")
    # Train on normal IDs
    trainer.train(normal_ids, epochs=50)

    # Predict on normal and anomalous images
    print("Predicting on normal images...")
    normal_predictions = trainer.predict(normal_ids)

    print("Predicting on anomalous images...")
    anomaly_predictions = trainer.predict(anomalous_ids)

    # Detailed analysis
    print("\n--- Prediction Results ---")
    print(f"Total Normal Images: {len(normal_ids)}")
    print(f"Correctly Identified Normal Images: {np.sum(normal_predictions)} / {len(normal_ids)}")

    print(f"\nTotal Anomalous Images: {len(anomalous_ids)}")
    print(f"Detected Anomalous Images: {len(anomalous_ids) - np.sum(anomaly_predictions)} / {len(anomalous_ids)}")

    # Optional: Feature extraction for further analysis
    print("\nExtracting features...")
    normal_features = trainer.extract_features(normal_ids)
    anomaly_features = trainer.extract_features(anomalous_ids)

    print("Normal features shape:", normal_features.shape)
    print("Anomaly features shape:", anomaly_features.shape)

    # Save model (optional)
    save_path = os.path.join('best_model_ResNet50_one_class.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'center': trainer.center,
        'radius': trainer.radius
    }, save_path)
    print(f"\nModel saved to {save_path}")


# Optional: Create a main block for direct script execution
if __name__ == '__main__':
    try:
        example_usage()
    except Exception as e:
        print(f"An error occurred: {e}")
