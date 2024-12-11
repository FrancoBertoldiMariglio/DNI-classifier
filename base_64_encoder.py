import os
import base64
from pathlib import Path


def convert_images_to_base64(input_dir, output_dir):
    """
    Convert all images in input_dir to base64 and save them as .txt files in output_dir.

    Args:
        input_dir (str): Path to directory containing images
        output_dir (str): Path where .txt files will be created
    """
    # Crear el directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extensiones de imagen comunes
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Procesar cada archivo en el directorio de entrada
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Verificar si es un archivo y tiene extensión de imagen
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                # Leer la imagen en modo binario
                with open(file_path, 'rb') as image_file:
                    # Codificar en base64
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                # Crear nombre para el archivo de salida
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_base64.txt")

                # Guardar la codificación en un archivo de texto
                with open(output_path, 'w') as txt_file:
                    txt_file.write(encoded_string)

                print(f"Convertido: {filename} -> {os.path.basename(output_path)}")

            except Exception as e:
                print(f"Error al procesar {filename}: {str(e)}")

# Ejemplo de uso
input_directory = "test_api_valid"
output_directory = "test_api_valid_base64"

convert_images_to_base64(input_directory, output_directory)