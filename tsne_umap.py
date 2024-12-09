import torch
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from autoencoder import DNIDataset, Encoder


def extract_embeddings(model_path, data_dir, batch_size=32, device='cuda'):
    # Load encoder
    encoder = Encoder().to(device)
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    # Load dataset
    dataset = DNIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embedding = encoder(batch)
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings)


def visualize_embeddings(embeddings, labels, method='tsne', **kwargs):
    if method == 'tsne':
        reducer = TSNE(n_components=2, **kwargs)
    else:
        reducer = umap.UMAP(n_components=2, **kwargs)

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    colors = ['blue' if label == 'normal' else 'red' for label in labels]
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                c=colors, alpha=0.5, label=['Normal', 'Anomaly'])
    plt.title(f'Embeddings visualization using {method.upper()}')
    plt.legend()
    plt.show()

# Update main() to pass labels to visualize_embeddings
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "models/dni_anomaly_detector.pt"
    normal_data_dir = "autoencoder_data/train_all"
    anomaly_data_dir = "Dni fotocopias (2)"

    print("Extracting normal embeddings...")
    normal_embeddings = extract_embeddings(model_path, normal_data_dir, device=device)
    print("Extracting anomaly embeddings...")
    anomaly_embeddings = extract_embeddings(model_path, anomaly_data_dir, device=device)

    all_embeddings = np.vstack([normal_embeddings, anomaly_embeddings])
    labels = np.array(['normal'] * len(normal_embeddings) + ['anomaly'] * len(anomaly_embeddings))

    print("Generating t-SNE visualization...")
    visualize_embeddings(all_embeddings, labels, method='tsne', perplexity=30, n_iter=1000)

    print("Generating UMAP visualization...")
    visualize_embeddings(all_embeddings, labels, method='umap', n_neighbors=15, min_dist=0.1)


if __name__ == "__main__":
    main()