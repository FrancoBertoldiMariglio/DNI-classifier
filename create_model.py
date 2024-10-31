import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

exp_config = dict()

# Configuración de dataset y división
exp_config['test_size'] = 0.2
exp_config['val_size'] = 0.2
exp_config['seed'] = 42

# Configuración del modelo
exp_config['input_size'] = (224, 224)  # Tamaño para ResNet
exp_config['n_channels'] = 3
exp_config['backbone'] = 'resnet50'

# Hiperparámetros de entrenamiento
exp_config['n_episodes'] = 200
exp_config['n_support'] = 5  # Muestras de soporte por clase
exp_config['n_query'] = 10   # Muestras de consulta por clase
exp_config['learning_rate'] = 1e-3
exp_config['weight_decay'] = 1e-4

# Configuración adicional
exp_config['device'] = 'mps' if torch.mps.is_available() else 'cpu'


class FewShotImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset for few-shot learning

        Args:
            image_paths (list): List of image file paths
            labels (list): Corresponding labels
            transform (callable, optional): Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._get_default_transforms()

    def _get_default_transforms(self):
        """
        Generate default image transformations
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class PrototypicalNetwork(nn.Module):
    def __init__(self, num_classes=2):
        """
        Prototypical Network for Few-Shot Learning

        Args:
            num_classes (int): Number of classes
        """
        super(PrototypicalNetwork, self).__init__()

        # Use ResNet50 as feature extractor
        backbone = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Learnable projection layers
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        """
        Forward pass for feature extraction

        Args:
            x (tensor): Input images

        Returns:
            tensor: Projected feature embeddings
        """
        features = self.feature_extractor(x).squeeze()
        return self.projection(features)

    def compute_prototypes(self, support_features, support_labels):
        """
        Compute class prototypes from support set

        Args:
            support_features (tensor): Feature embeddings of support set
            support_labels (tensor): Labels of support set

        Returns:
            dict: Prototype vectors for each class
        """
        prototypes = {}
        for label in torch.unique(support_labels):
            mask = support_labels == label
            prototypes[label.item()] = support_features[mask].mean(0)
        return prototypes

    def predict(self, query_features, prototypes):
        """
        Predict labels based on distance to prototypes

        Args:
            query_features (tensor): Feature embeddings of query set
            prototypes (dict): Prototype vectors for each class

        Returns:
            tensor: Predicted labels
        """
        predictions = []
        for feat in query_features:
            distances = {k: torch.norm(feat - proto) for k, proto in prototypes.items()}
            pred = min(distances, key=distances.get)
            predictions.append(pred)
        return torch.tensor(predictions)


class FewShotLearner:
    def __init__(self, image_dir, test_size=0.2, random_state=42):
        """
        Few-Shot Learning Training Pipeline

        Args:
            image_dir (str): Directory containing images
            test_size (float): Proportion of dataset for validation
            random_state (int): Random seed for reproducibility
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.image_paths, self.labels = self._load_images(image_dir)

        # Split dataset
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths, self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=random_state
        )

        # Create datasets
        self.train_dataset = FewShotImageDataset(train_paths, train_labels)
        self.val_dataset = FewShotImageDataset(val_paths, val_labels)

        # Model and optimizer
        self.model = PrototypicalNetwork().to(self.device)
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

    def _load_images(self, image_dir):
        """
        Load images and their labels from directory

        Args:
            image_dir (str): Directory containing images

        Returns:
            tuple: List of image paths and corresponding labels
        """
        image_paths = []
        labels = []
        for label, class_name in enumerate(['valid', 'invalid']):
            class_dir = os.path.join(image_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(label)
        return image_paths, labels

    def train_step(self, support_set, query_set):
        """
        Single training step with support and query sets

        Args:
            support_set (tuple): Support set images and labels
            query_set (tuple): Query set images and labels

        Returns:
            float: Training loss
        """
        support_images, support_labels = support_set
        query_images, query_labels = query_set

        support_images = support_images.to(self.device)
        query_images = query_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_labels = query_labels.to(self.device)

        self.optimizer.zero_grad()

        # Extract features
        support_features = self.model(support_images)
        query_features = self.model(query_images)

        # Compute prototypes
        prototypes = self.model.compute_prototypes(support_features, support_labels)

        # Predict labels
        predictions = self.model.predict(query_features, prototypes)

        # Compute loss
        loss = self.criterion(
            query_features,
            query_labels
        )
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes=100, n_support=5, n_query=10):
        """
        Training loop for few-shot learning

        Args:
            n_episodes (int): Number of training episodes
            n_support (int): Number of support samples per class
            n_query (int): Number of query samples per class
        """
        best_accuracy = 0
        best_model_path = 'best_model_ResNet50_few_shot.pth'

        for episode in range(n_episodes):
            # Sample support and query sets
            support_set = self._sample_episode(self.train_dataset, n_support, n_query)
            query_set = self._sample_episode(self.train_dataset, n_support, n_query)

            loss = self.train_step(support_set, query_set)

            # Evaluar cada cierto número de episodios
            if episode % 10 == 0:
                print(f"Episode {episode}, Loss: {loss:.4f}")

                # Evaluar el modelo actual
                current_accuracy = self.evaluate()
                print(f"Current Accuracy: {current_accuracy * 100:.2f}%")

                # Guardar el mejor modelo
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

        print(f"Best Model Saved at: {best_model_path}")

    def _sample_episode(self, dataset, n_support, n_query):
        """
        Sample support and query sets for an episode

        Args:
            dataset (Dataset): Input dataset
            n_support (int): Number of support samples per class
            n_query (int): Number of query samples per class

        Returns:
            tuple: Sampled images and labels
        """
        unique_labels = np.unique(dataset.labels)
        episode_images = []
        episode_labels = []

        for label in unique_labels:
            label_indices = np.where(np.array(dataset.labels) == label)[0]
            np.random.shuffle(label_indices)

            support_indices = label_indices[:n_support]
            query_indices = label_indices[n_support:n_support + n_query]

            for idx in support_indices:
                episode_images.append(dataset[idx][0])
                episode_labels.append(label)

            for idx in query_indices:
                episode_images.append(dataset[idx][0])
                episode_labels.append(label)

        return (
            torch.stack(episode_images),
            torch.tensor(episode_labels)
        )

    def evaluate(self, n_episodes=50, n_support=5, n_query=10):
        """
        Evaluate model performance

        Args:
            n_episodes (int): Number of evaluation episodes
            n_support (int): Number of support samples per class
            n_query (int): Number of query samples per class

        Returns:
            float: Average accuracy across episodes
        """
        accuracies = []
        for _ in range(n_episodes):
            support_set = self._sample_episode(self.val_dataset, n_support, n_query)
            query_set = self._sample_episode(self.val_dataset, n_support, n_query)

            support_images, support_labels = support_set
            query_images, query_labels = query_set

            # Ensure all tensors are on the same device
            support_images = support_images.to(self.device)
            query_images = query_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_labels = query_labels.to(self.device)

            support_features = self.model(support_images)
            query_features = self.model(query_images)

            prototypes = self.model.compute_prototypes(support_features, support_labels)
            predictions = self.model.predict(query_features, prototypes)

            # Ensure predictions and query_labels are on the same device
            predictions = predictions.to(self.device)

            accuracy = (predictions == query_labels).float().mean()
            accuracies.append(accuracy.item())

        return np.mean(accuracies)


# Usage example
def main():
    # Adjust the path to your image directory
    image_dir = 'dataset'

    # Initialize few-shot learner
    learner = FewShotLearner(image_dir)

    # Train the model
    learner.train(n_episodes=200)

    # Evaluate performance
    accuracy = learner.evaluate()
    print(f"Few-Shot Learning Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()