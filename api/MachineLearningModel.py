import os
import numpy as np
import tensorflow as tf
import joblib
from keras.api.models import load_model, Model
from keras.api.layers import Input, Dense, Reshape, UpSampling2D, Conv2D
from PIL import Image
import io
import keras


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
            self.encoder = load_model(model_path)
        else:
            # Si se pasa un objeto AnomalyDetector
            self.encoder = model_path.get_latent_model()

    def extract_features_from_image(self, image):
        """
        Extrae características de una imagen
        Args:
            image: Imagen PIL o array de numpy
        Returns:
            Características extraídas
        """
        # Convertir a array si es imagen PIL
        if isinstance(image, Image.Image):
            # Redimensionar la imagen
            image = image.resize((224, 224))
            # Convertir a array
            image = np.array(image)

        # Asegurar que la imagen tenga 3 canales
        if len(image.shape) == 2:  # Imagen en escala de grises
            image = np.stack((image,) * 3, axis=-1)

        # Añadir dimensión de batch
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Normalizar
        image = image.astype('float32') / 255.0

        # Preprocesar para ResNet50
        x_preprocessed = tf.keras.applications.resnet50.preprocess_input(image)

        # Extraer características
        encoded = self.encoder(x_preprocessed)
        return encoded.numpy()


class AnomalyDetectionModel:
    def __init__(self, autoencoder_path, svm_path):
        """
        Inicializa el modelo cargando el autoencoder y el SVM
        """
        # Configurar para usar CPU
        with tf.device('/CPU:0'):
            # Cargar autoencoder
            self.autoencoder = load_trained_model(autoencoder_path)
            self.feature_extractor = FeatureExtractor(self.autoencoder)

            # Cargar SVM
            self.svm = joblib.load(svm_path)

    def predict_image(self, image_data):
        """
        Realiza predicción sobre una imagen
        """
        with tf.device('/CPU:0'):
            # Manejar diferentes tipos de entrada
            if isinstance(image_data, (str, bytes, io.BytesIO)):
                if isinstance(image_data, str):
                    image = Image.open(image_data)
                else:
                    image = Image.open(io.BytesIO(image_data) if isinstance(image_data, bytes) else image_data)
            else:
                image = image_data

            # Extraer características
            features = self.feature_extractor.extract_features_from_image(image)

            # Obtener predicción y probabilidades
            prediction = self.svm.predict(features)[0]
            probabilities = self.svm.predict_proba(features)[0]

            # Crear respuesta
            confidence = float(probabilities[int(prediction)])
            result = {
                "prediction": "valid" if prediction == 0 else "anomaly",
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    "valid": round(float(probabilities[0]) * 100, 2),
                    "anomaly": round(float(probabilities[1]) * 100, 2)
                }
            }

            return result


def load_trained_model(model_path):
    """
    Carga un modelo entrenado y retorna una instancia de AnomalyDetector
    """
    autoencoder = AnomalyDetector()
    loaded_model = load_model(model_path)
    autoencoder.model.set_weights(loaded_model.get_weights())
    return autoencoder


# Ejemplo de uso:
"""
# Inicializar modelo
model = AnomalyDetectionModel(
    autoencoder_path='model_checkpoints/best_model_cropped_original.keras',
    svm_path='svm_classifier.joblib'
)

# Predecir desde archivo
result = model.predict_image('path/to/image.jpg')

# Predecir desde bytes
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()
result = model.predict_image(image_bytes)

print(result)
# Output:
# {
#     "prediction": "valid",
#     "confidence": 95.67,
#     "probabilities": {
#         "valid": 95.67,
#         "anomaly": 4.33
#     }
# }
"""