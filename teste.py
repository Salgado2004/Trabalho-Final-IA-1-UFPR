import cv2
import numpy as np
from joblib import load

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

def main():
    while True:
        print("Escolha uma imagem para testar:")
        for i, image in enumerate(IMAGES):
            print(f"{i + 1}: {image}")
        print("0: Sair")
        choice = int(input("Digite o número da imagem: ")) - 1
        if choice >= len(IMAGES):
            print("Escolha inválida.")
            return  
        if choice < 0:
            print("Saindo...")
            return
        selected_image = IMAGES[choice]
        print(f"Processando imagem: {selected_image}")
        data = prepare_data(selected_image)

        classified_data = classify_data(data)
        print("Resultados da classificação:")
        image = cv2.imread(selected_image)
        for item in classified_data:
            draw_class(image, item['contour'], item['class'])
        resized = cv2.resize(image, (0, 0), fx=0.23, fy=0.23)
        cv2.imshow("Classificacao", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\033[H\033[J", end="")

def apply_sobel(img):
    img_gray = img
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return img_sobel

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
        if  cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            if 1 <= w / h <= 4:
                contour = {"contour": c}
                cropped = enhanced[y:y+h, x:x+w]
                hist = cv2.calcHist([cropped], [0], None, [256], [0, 256]).flatten()
                total_pixels = np.sum(hist)
                media = np.sum(hist * np.arange(256)) / total_pixels
                desvioPadrao = np.sqrt(np.sum(hist * (np.arange(256) - media) ** 2) / total_pixels)
                cumsum = np.cumsum(hist)
                mediana = np.searchsorted(cumsum, total_pixels // 2)
                contour["features"] = [mediana, desvioPadrao]
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

main()