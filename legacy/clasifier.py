import cupy as cp
import cv2
from scipy import stats
from PIL import Image

def analyze_dni_image(image):
    """
    Analiza múltiples características de la imagen para detectar si es un DNI original
    o una impresión/fotocopia.

    :param image: Imagen PIL
    :return: Dict con métricas y análisis
    """
    # Convertir a escala de grises si no lo está
    gray_image = image.convert('L')
    img_array = cp.array(gray_image)

    # Convertir a CV2 para algunos análisis
    cv2_image = cp.array(gray_image)

    # 1. Análisis de patrones de impresión
    def detect_printer_patterns():
        # Detectar patrones regulares típicos de impresoras
        # Usar FFT para detectar patrones periódicos
        f = cp.fft.fft2(img_array)
        fshift = cp.fft.fftshift(f)
        magnitude_spectrum = 20 * cp.log(cp.abs(fshift))
        # Buscar picos regulares en el espectro de frecuencia
        periodic_score = cp.std(magnitude_spectrum)
        return periodic_score

    # 2. Análisis de bordes
    def analyze_edges():
        # Convertir CuPy array a NumPy array
        cv2_image_np = cp.asnumpy(cv2_image)
        edges = cv2.Canny(cv2_image_np, 100, 200)
        edge_density = cp.sum(edges > 0) / edges.size
        return edge_density

    # 3. Análisis de textura
    def analyze_texture():
        # GLCM (Gray-Level Co-Occurrence Matrix)
        glcm = cp.zeros((256, 256))
        rows, cols = img_array.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                glcm[img_array[i, j], img_array[i, j + 1]] += 1

        # Normalizar GLCM
        glcm = glcm / cp.sum(glcm)

        # Calcular características de textura
        contrast = cp.sum(cp.square(cp.arange(256)[:, None] - cp.arange(256)[None, :]) * glcm)
        homogeneity = cp.sum(glcm / (1 + cp.square(cp.arange(256)[:, None] - cp.arange(256)[None, :])))

        return contrast, homogeneity

    # 4. Análisis de ruido
    def analyze_noise():
        # Calcular la desviación estándar local
        local_std = cp.std(img_array)

        # Aplicar filtro de mediana y comparar con original
        median_filtered = cv2.medianBlur(cp.asnumpy(cv2_image), 3)
        noise_diff = cp.mean(cp.abs(cv2_image - cp.array(median_filtered)))

        return local_std, noise_diff

    # 5. Análisis de niveles de gris
    def analyze_gray_levels():
        histogram = cp.histogram(img_array, bins=256, range=(0, 256))[0]
        histogram = histogram / cp.sum(histogram)

        # Calcular entropía
        entropy = -cp.sum(histogram * cp.log2(histogram + 1e-10))

        # Calcular momentos estadísticos
        mean = cp.mean(img_array)
        std = cp.std(img_array)
        skewness = stats.skew(cp.asnumpy(img_array).ravel())
        kurtosis = stats.kurtosis(cp.asnumpy(img_array).ravel())

        return entropy, mean, std, skewness, kurtosis

    # Ejecutar los análisis una sola vez y almacenar resultados en variables
    printer_pattern_score = detect_printer_patterns()
    edge_density = analyze_edges()
    texture_contrast, texture_homogeneity = analyze_texture()
    noise_level, noise_difference = analyze_noise()
    gray_entropy, gray_mean, gray_std, gray_skewness, gray_kurtosis = analyze_gray_levels()

    # Crear el dict con los resultados
    metrics = {
        'printer_pattern_score': printer_pattern_score,
        'edge_density': edge_density,
        'texture_contrast': texture_contrast,
        'texture_homogeneity': texture_homogeneity,
        'noise_level': noise_level,
        'noise_difference': noise_difference,
        'gray_entropy': gray_entropy,
        'gray_mean': gray_mean,
        'gray_std': gray_std,
        'gray_skewness': gray_skewness,
        'gray_kurtosis': gray_kurtosis
    }

    return metrics

def classify_dni(image, thresholds=None):
    """
    Clasifica un DNI como original o impresión basándose en múltiples características.

    :param image: Imagen PIL del DNI
    :param thresholds: Dict con umbrales para cada métrica (opcional)
    :return: 'valida' o 'invalida' y métricas
    """
    # Analizar la imagen
    metrics = analyze_dni_image(image)

    # Umbrales por defecto (estos deberían ajustarse con un conjunto de entrenamiento)
    default_thresholds = {
        'printer_pattern_score': 50,
        'edge_density': 0.1,
        'texture_contrast': 100,
        'noise_level': 20,
        'gray_entropy': 7
    }

    thresholds = thresholds or default_thresholds

    # Sistema de puntuación
    score = 0

    # Reglas de clasificación invertidas
    if metrics['printer_pattern_score'] <= thresholds['printer_pattern_score']:
        score -= 1  # Patrones menos regulares sugieren un DNI original

    if metrics['edge_density'] >= thresholds['edge_density']:
        score -= 1  # Bordes más definidos sugieren un DNI original

    if metrics['texture_contrast'] >= thresholds['texture_contrast']:
        score -= 1  # Mayor contraste de textura sugiere un DNI original

    if metrics['noise_level'] >= thresholds['noise_level']:
        score -= 1  # Mayor nivel de ruido sugiere un DNI original

    if metrics['gray_entropy'] >= thresholds['gray_entropy']:
        score -= 1  # Mayor entropía sugiere un DNI original

    # Clasificación final invertida
    result = "invalida" if score <= -3 else "valida"

    return result, metrics


if __name__ == '__main__':
    # image_path_valid: str = 'DNI-Validos/Documento_Nacional_Identidad_Dorso_Versión_1_1a9ddb03-b8f1-47dd-aae4-b1d7f2627b3c.jpeg'
    # image_valid = Image.open(image_path_valid)
    # result, _ = classify_dni(image_valid)
    # print(result)

    image_path_invalid = '../DNI-Invalidos/IMG_20241023_122058232.jpg'
    image_invalid = Image.open(image_path_invalid)
    result, _ = classify_dni(image_invalid)
    print(result)
