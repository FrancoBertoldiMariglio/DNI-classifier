# predictor/model.py
import base64
import io
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Union
from api.RelationNet import EmbeddingNet, RelationModule

logger = logging.getLogger(__name__)


class RelationNetPredictor:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        try:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")

            self.embedding_net = EmbeddingNet().to(self.device)
            self.relation_module = RelationModule().to(self.device)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.embedding_net.load_state_dict(checkpoint['embedding_state_dict'])
            self.relation_module.load_state_dict(checkpoint['relation_state_dict'])

            self.embedding_net.eval()
            self.relation_module.eval()

            self.transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def load_image_from_path(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading image from {image_path}: {str(e)}")

    def load_image_from_base64(self, base64_str: str) -> torch.Tensor:
        try:
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {str(e)}")

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embedding_net(image_tensor)

    def predict(self, query_base64: str, support_folder: str) -> float:
        """Core prediction logic"""
        try:
            # Load and validate support images
            support_path = Path(support_folder)
            valid_images = list(support_path.glob('valid/*.jpg')) + list(support_path.glob('valid/*.png'))
            invalid_images = list(support_path.glob('invalid/*.jpg')) + list(support_path.glob('invalid/*.png'))

            if not valid_images and not invalid_images:
                raise ValueError(f"No support images found in {support_folder}")

            # Prepare support set
            support_images = valid_images + invalid_images
            support_labels = [1] * len(valid_images) + [0] * len(invalid_images)

            # Get query embedding
            query_tensor = self.load_image_from_base64(query_base64)
            query_features = self.get_embedding(query_tensor)

            # Process support set
            support_features = []
            for img_path in support_images:
                img_tensor = self.load_image_from_path(img_path)
                features = self.get_embedding(img_tensor)
                support_features.append(features)

            support_features = torch.cat(support_features)
            support_labels = torch.tensor(support_labels, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                # Check for exact matches
                similarities = [
                    F.cosine_similarity(
                        query_features.view(1, -1),
                        feat.view(1, -1)
                    ).item()
                    for feat in support_features
                ]

                max_similarity = max(similarities)
                if max_similarity > 0.95:
                    idx = similarities.index(max_similarity)
                    if support_labels[idx] == 0:
                        return 0.0

                # Regular prediction
                n_support = support_features.size(0)
                query_features_ext = query_features.unsqueeze(1).expand(-1, n_support, -1, -1, -1)
                support_features_ext = support_features.unsqueeze(0).expand(1, -1, -1, -1, -1)

                relation_pairs = torch.cat([query_features_ext, support_features_ext], dim=2)
                relation_pairs = relation_pairs.reshape(-1, *relation_pairs.shape[2:])

                relations = self.relation_module(relation_pairs)
                relations = relations.view(1, n_support)

                weighted_relations = relations * support_labels
                prediction = weighted_relations.mean(dim=1)
                return torch.sigmoid(prediction).item()

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise