import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import umap
import wandb
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.api.models import save_model, load_model, Model
from keras.api.layers import Input, Dense, Reshape, UpSampling2D, Conv2D
import tensorflow as tf
import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def setup_device():
    """
    Configure device settings (CPU/GPU)
    """
    # Intentar usar GPU si está disponible
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"GPU(s) found: {physical_devices}")
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            return "GPU"
    except:
        pass
    print("Using CPU")
    return "CPU"

class AnomalyDetector(Model):
    def __init__(self, input_dim=224, latent_dim=256):
        super(AnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Crear el encoder basado en ResNet50
        resnet = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(input_dim, input_dim, 3),
            pooling='avg'
        )

        # Congelar las capas de ResNet50
        for layer in resnet.layers:
            layer.trainable = False

        # Input layer
        input_layer = Input(shape=(input_dim, input_dim, 3))

        # Encoder path
        x = resnet(input_layer)
        latent_view = Dense(latent_dim, activation='relu', name='latent_space')(x)

        # Decoder path
        x = Dense(7 * 7 * 512, activation='relu')(latent_view)
        x = Reshape((7, 7, 512))(x)

        # Upsampling blocks
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2, 2))(x)
        output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        # Crear modelos
        self.encoder_model = Model(input_layer, latent_view, name='encoder')
        self.decoder_model = Model(latent_view, output_layer, name='decoder')
        self.model = Model(input_layer, output_layer, name='autoencoder')

    def preprocess_input(self, x):
        """Preprocesar imágenes para ResNet50"""
        x = tf.keras.applications.resnet50.preprocess_input(x)
        return x

    def encode(self, x):
        """Obtener representación del espacio latente"""
        x_preprocessed = self.preprocess_input(x)
        return self.encoder_model(x_preprocessed)

    def decode(self, z):
        """Reconstruir desde el espacio latente"""
        return self.decoder_model(z)

    def call(self, x):
        x_preprocessed = self.preprocess_input(x)
        encoded = self.encoder_model(x_preprocessed)
        decoded = self.decoder_model(encoded)
        return decoded

    def get_latent_model(self):
        """Retorna el modelo encoder para extracción de características"""
        return self.encoder_model


class FeatureExtractor:
    def __init__(self, model_path=None):
        if isinstance(model_path, str):
            # Si se pasa un path, cargar solo el encoder
            self.encoder = keras.models.load_model(model_path)
        else:
            # Si se pasa un objeto AnomalyDetector
            self.encoder = model_path.get_latent_model()

    def extract_features(self, data_dir):
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=(224, 224),
            batch_size=32,
            shuffle=False,
            label_mode=None
        )
        dataset = dataset.map(lambda x: x / 255.0)

        features = []
        with tf.device('/CPU:0'):
            for batch in dataset:
                # Preprocesar las imágenes
                x_preprocessed = tf.keras.applications.resnet50.preprocess_input(batch)
                encoded = self.encoder(x_preprocessed)
                features.append(encoded.numpy())

        return np.vstack(features)


def load_trained_model(model_path):
    """
    Carga un modelo entrenado y retorna el encoder y el modelo completo
    """
    # Crear una nueva instancia del modelo
    autoencoder = AnomalyDetector()

    # Cargar los pesos del modelo guardado
    loaded_model = keras.models.load_model(model_path)
    autoencoder.model.set_weights(loaded_model.get_weights())

    return autoencoder


# Función de utilidad para visualizar reconstrucciones
def plot_reconstructions(autoencoder, test_images, n=5):
    import matplotlib.pyplot as plt

    # Obtener reconstrucciones
    reconstructions = autoencoder(test_images[:n])

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i])
        plt.title('Original')
        plt.axis('off')

        # Reconstrucción
        plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructions[i])
        plt.title('Reconstrucción')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def create_visualization_dataset(feature_extractor, valid_dir, anomaly_dir):
    # Extract features for both classes
    valid_features = feature_extractor.extract_features(valid_dir)
    anomaly_features = feature_extractor.extract_features(anomaly_dir)

    # Create labels
    valid_labels = np.zeros(len(valid_features))
    anomaly_labels = np.ones(len(anomaly_features))

    # Combine features and labels
    X = np.vstack([valid_features, anomaly_features])
    y = np.hstack([valid_labels, anomaly_labels])

    return X, y


def visualize_embeddings(X, y, title_prefix=""):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'{title_prefix} UMAP Visualization',
                                        f'{title_prefix} t-SNE Visualization'))

    # UMAP
    reducer_umap = umap.UMAP(random_state=42)
    embedding_umap = reducer_umap.fit_transform(X)

    # t-SNE
    reducer_tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = reducer_tsne.fit_transform(X)

    # Plot UMAP
    for label, name in [(0, 'Valid'), (1, 'Anomaly')]:
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=embedding_umap[mask, 0],
                y=embedding_umap[mask, 1],
                mode='markers',
                name=f'{name} (UMAP)',
                marker=dict(size=8)
            ),
            row=1, col=1
        )

    # Plot t-SNE
    for label, name in [(0, 'Valid'), (1, 'Anomaly')]:
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=embedding_tsne[mask, 0],
                y=embedding_tsne[mask, 1],
                mode='markers',
                name=f'{name} (t-SNE)',
                marker=dict(size=8)
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        height=500,
        width=1000,
        title_text=f"{title_prefix} Dimensionality Reduction Visualization",
        showlegend=True
    )

    # Log plot en wandb
    wandb.log({f"{title_prefix}_embeddings": fig})

    return fig


def train_autoencoder(train_data, validation_data=None, epochs=20):
    """
    Train the autoencoder model with tf.data.Dataset and model checkpointing
    """
    # Log autoencoder config
    wandb.config.update({
        "learning_rate": 0.001,
        "epochs": epochs,
        "batch_size": 32,
        "architecture": "ResNet50_Autoencoder",
        "optimizer": "Adam",
        "loss": "MSE",
        "device": "GPU"
    }, allow_val_change=True)

    # Configurar la estrategia de entrenamiento
    strategy = tf.distribute.OneDeviceStrategy("/CPU:0")

    with tf.device('/CPU:0'):
        autoencoder = AnomalyDetector()

        # Compilar modelo con opciones ajustadas
        autoencoder.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            run_eagerly=False,  # Cambiar a False para mejor rendimiento
            jit_compile=False  # Desactivar JIT compilation
        )

        # Crear directorio para checkpoints si no existe
        checkpoint_dir = 'model_checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Configurar callbacks
        callbacks = [
            # Guardar el mejor modelo
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model_cropped_original.keras'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            # Early stopping para evitar overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir learning rate cuando el entrenamiento se estanca
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # CSV Logger para tener un backup local del progreso
            keras.callbacks.CSVLogger(
                os.path.join(checkpoint_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]

        # Modificar los datasets para que sean input=output
        train_ds_auto = train_data.map(lambda x: (x, x)).prefetch(tf.data.AUTOTUNE)
        val_ds_auto = validation_data.map(lambda x: (x, x)).prefetch(
            tf.data.AUTOTUNE) if validation_data is not None else None

        try:
            # Entrenar modelo
            history = autoencoder.model.fit(
                train_ds_auto,
                epochs=epochs,
                validation_data=val_ds_auto,
                callbacks=callbacks,
                verbose=1
            )

            # Cargar los mejores pesos
            best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')
            if os.path.exists(best_model_path):
                print("Loading best model weights...")
                autoencoder = keras.models.load_model(best_model_path)

            # Log del mejor valor de pérdida
            best_val_loss = min(history.history['val_loss'])
            wandb.log({
                'best_val_loss': best_val_loss,
                'final_val_loss': history.history['val_loss'][-1]
            })

            return autoencoder, history

        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            print("Intentando entrenar en CPU...")

            # Fallback a CPU si hay error en GPU
            with tf.device('/CPU:0'):
                history = autoencoder.model.fit(
                    train_ds_auto,
                    epochs=epochs,
                    validation_data=val_ds_auto,
                    callbacks=callbacks,
                    verbose=1
                )
                return autoencoder, history


def train_svm_with_gridsearch(X, y):
    """
    Train SVM classifier with GridSearch optimization and WandB logging

    Args:
        X: Features array
        y: Labels array

    Returns:
        best_pipeline: Trained pipeline with best parameters
        test_data: Tuple of (X_test, y_test)
    """
    # Create pipeline
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])

    # Define parameter grid
    param_grid = {
        'svm__kernel': ['rbf', 'linear'],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
        'svm__class_weight': [None, 'balanced']
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_pipeline = grid_search.best_estimator_

    # Make predictions with best model
    y_pred = best_pipeline.predict(X_test)

    # Calculate metrics
    metrics = classification_report(y_test, y_pred,
                                    target_names=['Valid', 'Anomaly'],
                                    output_dict=True)

    # Log metrics and parameters to wandb
    wandb.log({
        "accuracy": metrics['accuracy'],
        "valid_precision": metrics['Valid']['precision'],
        "valid_recall": metrics['Valid']['recall'],
        "valid_f1": metrics['Valid']['f1-score'],
        "anomaly_precision": metrics['Anomaly']['precision'],
        "anomaly_recall": metrics['Anomaly']['recall'],
        "anomaly_f1": metrics['Anomaly']['f1-score'],

        # Log best parameters
        "best_kernel": grid_search.best_params_['svm__kernel'],
        "best_C": grid_search.best_params_['svm__C'],
        "best_gamma": grid_search.best_params_['svm__gamma'],
        "best_class_weight": grid_search.best_params_['svm__class_weight'],

        # Log cross-validation scores
        "cv_mean_score": grid_search.best_score_,
        "cv_std_score": np.std(grid_search.cv_results_['split0_test_score']),
    })

    # Print results
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("\nBest cross-validation score:", grid_search.best_score_)
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Valid', 'Anomaly']))

    # Create and log confusion matrix plot
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Valid', 'Anomaly'],
                    yticklabels=['Valid', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
    except Exception as e:
        print(f"Could not create confusion matrix plot: {e}")

    return best_pipeline, (X_test, y_test)


def main(train_auto=True, train_support=True):
    # Iniciar una única sesión de wandb
    wandb.init(
        project='anomaly_detection',
        name='full_training_pipeline',
        config={
            "pipeline_steps": {
                "train_autoencoder": train_auto,
                "train_svm": train_support
            }
        }
    )

    try:
        # Configurar dispositivo
        device = setup_device()
        wandb.config.update({"device": device})

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

        if train_auto:
            autoencoder, history = train_autoencoder(train_ds, val_ds, epochs=20)
        else:
            autoencoder = load_trained_model('api/best_model_cropped_original.keras')

        # Create feature extractor
        feature_extractor = FeatureExtractor(autoencoder)

        # Create dataset for visualization and classification
        X, y = create_visualization_dataset(
            feature_extractor,
            'svm_data/validTrain',
            'svm_data/invalidTrain'
        )

        # Visualize embeddings
        fig = visualize_embeddings(X, y, "Encoded Features")
        fig.show()

        if train_support:
            svm_model, (X_test, y_test) = train_svm_with_gridsearch(X, y)
            joblib.dump(svm_model, 'api/svm_classifier.joblib')
            wandb.save('api/svm_classifier.joblib')

        else:
            svm_model = joblib.load('api/svm_classifier.joblib')

        y_pred = svm_model.predict(X_test)
        fig_test = visualize_embeddings(X_test, y_pred, "SVM Predictions")
        fig_test.show()

    finally:
        # Cerrar wandb
        wandb.finish()

if __name__ == '__main__':
    main(train_auto=False)