import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import umap

import tensorflow as tf

# Ajustar los imports originales
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.api.datasets import fashion_mnist
from keras.api.models import Model
from keras.src.utils.image_dataset_utils import image_dataset_from_directory

from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Configuración de GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_umap(valid_dir, anomaly_dir):
    test_datagen = image_dataset_from_directory(rescale=1. / 255)

    # Generador para imágenes válidas
    valid_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )

    # Generador para anomalías
    anomaly_generator = test_datagen.flow_from_directory(
        anomaly_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )

    # Obtener batches
    valid_batch = next(valid_generator)
    anomaly_batch = next(anomaly_generator)

    # Codificar usando GPU
    with tf.device('/GPU:0'):
        valid_encoded = autoencoder.encoder(valid_batch)
        anomaly_encoded = autoencoder.encoder(anomaly_batch)

    # Combinar datos
    all_encoded = np.vstack([valid_encoded, anomaly_encoded])
    labels = np.array(['Valid'] * len(valid_encoded) + ['Anomaly'] * len(anomaly_encoded))

    # UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_encoded)

    # Plotly con colores diferentes por clase
    fig = go.Figure()
    for label in ['Valid', 'Anomaly']:
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            mode='markers',
            name=label,
            marker=dict(size=8)
        ))

    fig.update_layout(
        title='UMAP Visualization',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2'
    )

    return fig


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    roc_auc = roc_auc_score(y_test, pred_proba)
    print('confusion matrix')
    print(confusion)

    # ROC-AUC print
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    return confusion


class AnomalyDetector(Model):
    def __init__(self, input_dim=224, latent_dim=32):
        super(AnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim, input_dim)),
            layers.Reshape((input_dim, 224)),  # Reshape to 3D for Conv1D
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
        ])
        # Previously, I was using UpSampling. I am trying Transposed Convolution this time around.
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(input_dim)
        ])

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded



# Configurar datasets
train_ds = image_dataset_from_directory(
    'autoencoder_data/train_all',
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=123,
    shuffle=True,
    label_mode=None
)

val_ds = image_dataset_from_directory(
    'autoencoder_data/train_all',
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=123,
    shuffle=True,
    label_mode=None
)

# Normalizar y configurar pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x: x / 255.0).cache().prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x: x / 255.0).cache().prefetch(AUTOTUNE)

# Crear el modelo
with tf.device('/GPU:0'):
    autoencoder = AnomalyDetector()


def train():
    # Compilar modelo
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    # Entrenar
    with tf.device('/GPU:0'):
        history = autoencoder.fit(
            train_ds,
            epochs=20,
            batch_size=32,
            validation_data=val_ds,
            shuffle=True
        )

    # Visualizar pérdida
    plt.figure(figsize=(10, 8))
    sns.set_theme(font_scale=2)
    sns.set_style("white")
    plt.plot(history.history["loss"], label="Training Loss", linewidth=3.0)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=3.0)
    plt.legend()
    plt.show()


def test(dir):
    test_ds = image_dataset_from_directory(
        dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
        label_mode=None
    )
    test_ds = test_ds.map(lambda x: x / 255.0).cache()

    for images in test_ds.take(1):
        # Generar predicciones
        with tf.device('/GPU:0'):
            encoded_imgs = autoencoder.encoder(images)
            decoded_imgs = autoencoder.decoder(encoded_imgs)

        # Visualizar
        for i in range(3):
            plt.figure(figsize=(10, 8))
            sns.set_theme(font_scale=2)
            sns.set_style("white")
            plt.plot(images[i].numpy().flatten(), 'black', linewidth=2)
            plt.plot(decoded_imgs[i].numpy().flatten(), 'red', linewidth=2)
            plt.fill_between(range(len(images[i].numpy().flatten())),
                             decoded_imgs[i].numpy().flatten(),
                             images[i].numpy().flatten(),
                             color='lightcoral')
            plt.legend(labels=["Input", "Reconstruction", "Error"])
            plt.show()


if __name__ == '__main__':
    train()
    test('test/original/valid')
    fig = get_umap('test/original/valid', 'test/original/invalid')
    plt.show(fig)