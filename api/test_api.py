import os
import requests
import pandas as pd
from datetime import datetime
import logging


def process_directory(input_dir, output_file, api_url="http://0.0.0.0:8000/api/v1/predict"):
    """
    Process all base64 txt files in a directory, send them to API and save results

    Args:
        input_dir (str): Directory containing base64 txt files
        output_file (str): Path to save the CSV output
        api_url (str): URL of the prediction API
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results = []

    if not os.path.exists(input_dir):
        logging.error(f"El directorio {input_dir} no existe")
        return

    total_files = len([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    processed = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    base64_content = f.read().strip()

                # Formato correcto del payload
                payload = {
                    "base64_image": base64_content
                }

                response = requests.post(api_url, json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    result = {
                        'filename': filename,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'success',
                        'prediction': response_data.get('prediction'),
                        'confidence': response_data.get('confidence', None),
                        'error': None
                    }
                else:
                    result = {
                        'filename': filename,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'error',
                        'prediction': None,
                        'confidence': None,
                        'error': f'HTTP {response.status_code}: {response.text}'
                    }

                results.append(result)
                processed += 1
                logging.info(f'Procesado {processed}/{total_files}: {filename}')

            except Exception as e:
                logging.error(f'Error procesando {filename}: {str(e)}')
                results.append({
                    'filename': filename,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'error',
                    'prediction': None,
                    'confidence': None,
                    'error': str(e)
                })

    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logging.info(f'Resultados guardados en {output_file}')

        success_count = len(df[df['status'] == 'success'])
        error_count = len(df[df['status'] == 'error'])
        logging.info(f"""
        Resumen del procesamiento:
        - Total archivos: {total_files}
        - Exitosos: {success_count}
        - Errores: {error_count}
        """)

    except Exception as e:
        logging.error(f'Error guardando resultados: {str(e)}')


# Uso directo
if __name__ == "__main__":
    input_directory = "test_api_valid_base64"  # Cambiar esto
    output_csv = "resultados_validas.csv"  # Cambiar esto

    process_directory(input_directory, output_csv)