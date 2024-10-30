import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tools.utils import load_images
from tools.models import ResNet18
from tools.train_val_test import run

# Variables y rutas
valid_images_path = 'DNI-Validos'
invalid_images_path = 'DNI-Invalidos'

exp_config = dict()

test_size = 0.2
val_size = 0.2
seed = 42
input_size = (224, 224)  # Tamaño de entrada de ResNet18
n_channels = 3
batch_size = 64
num_epochs = 10
learning_rate = 0.001

exp_config['test_size'] = test_size
exp_config['val_size'] = val_size
exp_config['seed'] = seed
exp_config['input_size'] = input_size
exp_config['n_channels'] = n_channels
exp_config['batch_size'] = batch_size
exp_config['num_epochs'] = num_epochs
exp_config['learning_rate'] = learning_rate

# Cargar las imágenes
valid_images_dataset = load_images(valid_images_path, '1')  # Etiqueta '1' para válidos
invalid_images_dataset = load_images(invalid_images_path, '0')  # Etiqueta '0' para inválidos

# Unir ambos datasets en uno solo
mydataset = pd.concat([valid_images_dataset, invalid_images_dataset], ignore_index=True)

# Dividir en conjuntos de entrenamiento, validación y prueba con estratificación
train_val_df, test_df = train_test_split(mydataset, test_size=test_size, stratify=mydataset['etiqueta'], random_state=seed)
train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['etiqueta'], random_state=seed)


# Definir la clase del dataset
class DniDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Dataset para cargar imágenes y sus etiquetas desde un DataFrame.
        :param df: DataFrame que contiene los paths de las imágenes y sus etiquetas.
        :param transform: Transformaciones que se aplicarán a las imágenes.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = float(self.df.iloc[idx]['etiqueta'])  # Convertir a int

        # Cargar la imagen
        image = Image.open(img_path).convert("RGB")

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)

        # Convertir la etiqueta a tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización para ResNet
])

# Crear datasets para entrenamiento, validación y prueba
train_dataset = DniDataset(train_df, transform=transform)
val_dataset = DniDataset(val_df, transform=transform)
test_dataset = DniDataset(test_df, transform=transform)

# Crear DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# Cargar el modelo ResNet18 preentrenado
model = ResNet18()

# Definir la función de pérdida y el optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

exp_config['criterion'] = 'BCEWithLogitsLoss'
exp_config['optimizer'] = 'Adam'

model_name = "ResNet18_dni_adjusted"
exp_config['model_name'] = 'model_name'

run(train_dataloader, val_dataloader, test_dataloader, model, criterion, optimizer, model_name, num_epochs=num_epochs)
