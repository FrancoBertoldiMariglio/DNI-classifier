import joblib
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from pyglet import model
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from autoencoder import DNIDataset, Encoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score


def train_and_save_svm(embeddings, labels, save_path='models/svm_classifier.joblib'):
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(embeddings, labels)
    joblib.dump(svm, save_path)
    return svm

def load_svm(model_path='models/svm_classifier.joblib'):
    """Load trained SVM model from joblib file"""
    try:
        svm = joblib.load(model_path)
        return svm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def plot_decision_boundary(embeddings, labels, svm, method='tsne'):
    # Convert string labels to numeric
    numeric_labels = np.array([0 if label == 'normal' else 1 for label in labels])

    reducer = TSNE(n_components=2, max_iter=1000) if method == 'tsne' else umap.UMAP(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    svm_2d = SVC(kernel='rbf', random_state=42)
    svm_2d.fit(reduced_embeddings, numeric_labels)

    x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
    y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, levels=np.linspace(-1, 1, 3))
    colors = ['blue' if label == 'normal' else 'red' for label in labels]
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.8)
    plt.title(f'SVM Decision Boundary using {method.upper()} (2D projection)')
    plt.legend(['Normal', 'Anomaly'])
    plt.show()

def extract_embeddings(model_path, data_dir, batch_size=32, device='cuda'):
    encoder = Encoder().to(device)
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    dataset = DNIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embedding = encoder(batch)
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings)


def evaluate_classifiers(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    classifiers = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, clf in classifiers.items():
        print(f"\n{name} Results:")

        # Train and evaluate
        clf.fit(X_train, y_train)
        print("\nClassification Report:")
        print(classification_report(y_test, clf.predict(X_test)))

        # Cross validation
        cv_scores = cross_val_score(clf, embeddings, labels, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")


def visualize_embeddings(embeddings, labels, method='tsne', **kwargs):
    reducer = TSNE(n_components=2, **kwargs) if method == 'tsne' else umap.UMAP(n_components=2, **kwargs)
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    colors = ['blue' if label == 'normal' else 'red' for label in labels]
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.5)
    plt.title(f'Embeddings visualization using {method.upper()}')
    plt.legend(['Normal', 'Anomaly'])
    plt.show()


def train_svm_with_validation(encoder_path, train_normal_dir, train_anomaly_dir,
                              val_normal_dir, val_anomaly_dir, device='cuda'):
    # Extract training embeddings
    train_normal_emb = extract_embeddings(encoder_path, train_normal_dir, device=device)
    train_anomaly_emb = extract_embeddings(encoder_path, train_anomaly_dir, device=device)

    train_embeddings = np.vstack([train_normal_emb, train_anomaly_emb])
    train_labels = np.array(['normal'] * len(train_normal_emb) + ['anomaly'] * len(train_anomaly_emb))

    # Train SVM
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(train_embeddings, train_labels)
    joblib.dump(svm, 'models/svm_classifier.joblib')

    # Extract validation embeddings
    val_normal_emb = extract_embeddings(encoder_path, val_normal_dir, device=device)
    val_anomaly_emb = extract_embeddings(encoder_path, val_anomaly_dir, device=device)

    val_embeddings = np.vstack([val_normal_emb, val_anomaly_emb])
    val_labels = np.array(['normal'] * len(val_normal_emb) + ['anomaly'] * len(val_anomaly_emb))

    # Evaluate
    val_pred = svm.predict(val_embeddings)
    print("\nValidation Results:")
    print(classification_report(val_labels, val_pred))

    return svm, val_embeddings, val_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_path = "models/dni_anomaly_detector.pt"

    # Training and validation paths
    train_normal_dir = "autoencoder_data/train_all"
    train_anomaly_dir = "svm_data/invalidTrain"
    val_normal_dir = "test/valid"
    val_anomaly_dir = "svm_data/invalidVal"

    print("Training SVM and evaluating on validation set...")
    svm, val_embeddings, val_labels = train_svm_with_validation(
        encoder_path, train_normal_dir, train_anomaly_dir,
        val_normal_dir, val_anomaly_dir, device
    )

    print("\nGenerating visualizations for validation set...")
    visualize_embeddings(val_embeddings, val_labels, method='tsne', perplexity=30, n_iter=1000)
    visualize_embeddings(val_embeddings, val_labels, method='umap', n_neighbors=15, min_dist=0.1)

    plot_decision_boundary(val_embeddings, val_labels, svm, 'tsne')
    plot_decision_boundary(val_embeddings, val_labels, svm, 'umap')


if __name__ == "__main__":
    main()