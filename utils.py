import cv2
import numpy as np

IMAGES = [
    "./data/bom_1.jpg",
    "./data/bom_2.jpg",
    "./data/bom_3.jpg",
    "./data/bom_4.jpg",
    "./data/ruim_1.jpg",
    "./data/ruim_2.jpg",
    "./data/ruim_3.jpg",
    "./data/ruim_4.jpg"
] 

SEED = 1.9898

def apply_sobel(img):
    img_gray = img
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return img_sobel

def reveal_edges(image):
    sobel = apply_sobel(image)
    edges = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return edges

def prepare_image(image):
    temp = cv2.bitwise_not(image)
    mask = cv2.threshold(temp, 174, 255, cv2.THRESH_BINARY)[1]
    bean_no_bg = cv2.bitwise_and(image, image, mask=mask)
    bean_no_bg[bean_no_bg == 0] = 255

    return bean_no_bg

def extract_features(image):
    # Histograma e estatísticas básicas
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = np.sum(hist)
    media = np.sum(hist * np.arange(256)) / total_pixels
    desvioPadrao = np.sqrt(np.sum(hist * (np.arange(256) - media) ** 2) / total_pixels)
    cumsum = np.cumsum(hist)
    mediana = np.searchsorted(cumsum, total_pixels // 2)
    
    # Entropia
    prob = hist / total_pixels
    prob = prob[prob > 0]
    entropia = -np.sum(prob * np.log2(prob))

    # Bordas para silhueta
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        razao_largura_altura = w / h if h > 0 else 0
        circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
    else:
        area = perimetro = razao_largura_altura = circularidade = 0

    return (
        mediana * SEED,
        desvioPadrao * SEED,
        entropia,
        razao_largura_altura,
        circularidade
    )