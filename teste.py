import cv2
import os
import numpy as np
from joblib import load
from utils import apply_sobel, prepare_image, extract_features, IMAGES

def main():
    for image_path in IMAGES:
        print(f"Processando imagem: {image_path}")
        data = prepare_data(image_path)
        classified_data = classify_data(data)
        print("Salvando resultados da classificação")
        image = cv2.imread(image_path)
        for item in classified_data:
            draw_class(image, item['contour'], item['class'])
        save_results(image, image_path.removeprefix('./data/'))

def prepare_data(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao ler a imagem: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (11, 11), 0)
    sobel = apply_sobel(blurred)
    edges = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    data = []
    for c in contours:
        if  cv2.contourArea(c) > 750:
            x, y, w, h = cv2.boundingRect(c)
            if 1 <= w / h <= 4:
                bean_cropped = enhanced[y:y+h, x:x+w]

                bean_no_bg = prepare_image(bean_cropped)

                contour = {"contour": c}
                contour["features"] = list(extract_features(bean_no_bg))
                data.append(contour)
    return data
                
def classify_data(data):
    classificador = load('./out/classifier.joblib')
    for item in data:
        features = item['features']
        item['class'] = int(classificador.predict([features])[0])
    return data

def draw_class(image, contour, classe):
    color = (0, 255, 0) if classe == 1 else (0, 0, 255)
    cv2.drawContours(image, [contour], -1, color, 3)

def save_results(data, image_path):
    if not os.path.exists('./snapshots'):
        os.makedirs('./snapshots')
    base_name = os.path.basename(image_path)
    output_path = os.path.join('./snapshots', base_name)
    cv2.imwrite(output_path, data)

main()