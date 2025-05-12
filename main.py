import cv2
import numpy as np

# Classe 1 - Bom, 0 - Ruim
IMAGES = [
    { 'path': "./data/bom_1.jpg", 'classe': 1},
    { 'path': "./data/bom_2.jpg", 'classe': 1},
    { 'path': "./data/bom_3.jpg", 'classe': 1},
    { 'path': "./data/bom_4.jpg", 'classe': 1},
    { 'path': "./data/ruim_1.jpg", 'classe': 0},
    { 'path': "./data/ruim_2.jpg", 'classe': 0},
    { 'path': "./data/ruim_3.jpg", 'classe': 0},
    { 'path': "./data/ruim_4.jpg", 'classe': 0}
]

def main():
    for i, image in enumerate(IMAGES):
        print(f"Processando imagem: {image['path']}")
        preprocess_image(i, image['path'], image['classe'])

    cv2.destroyAllWindows()

def preprocess_image(id, image_path, classe):
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
    
    for i, contour in enumerate(contours):
        if  cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            if 1 <= w / h <= 4:
                cropped = enhanced[y:y+h, x:x+w]
                output_path = f"./out/{classe}-bean-{id*i}.jpg"
                cv2.imwrite(output_path, cropped)

def apply_sobel(img):
    img_gray = img
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return img_sobel

main()