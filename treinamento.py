import os
import cv2
import time
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

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
    #cleanup()
    if not os.path.exists("./out"):
        for i, image in enumerate(IMAGES):
            print(f"Processando imagem: {image['path']}")
            preprocess_image(i, image['path'], image['classe'])
        cv2.destroyAllWindows()

    if not os.path.exists("./bin"):
        print("Gerando arquivos .dat...")
        generate_dat_files()

    with open('./bin/beans_train.dat', 'r') as f:
        treinamento = []
        treinamento_classes = []
        for line in f:
            parts = line.strip().split("   ")
            if len(parts) < 3:
                continue
            features = list(map(float, parts[:-1]))
            label = int(parts[-1])
            treinamento.append(features)
            treinamento_classes.append(label)
    with open('./bin/beans_test.dat', 'r') as f:
        teste = []
        teste_classes = []
        for line in f:
            parts = line.strip().split("   ")
            if len(parts) < 3:
                continue
            features = list(map(float, parts[:-1]))
            label = int(parts[-1])
            teste.append(features)
            teste_classes.append(label)

    print("Iniciando treinamento do Perceptron...")
    start = time.time()
    buildMLPerceptron(treinamento, teste, treinamento_classes, teste_classes)
    end = time.time()
    print(f"Tempo de execução {end - start:.2f} segundos")    

def cleanup():
    if os.path.exists("./out"):
        for file in os.listdir("./out"):
            os.remove(os.path.join("./out", file))
        os.rmdir("./out")
    
    os.makedirs("./out")

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

def generate_dat_files():
    os.makedirs("./bin")

    images = []
    for file in os.listdir("./out"):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join("./out", file))
            if img is not None:
                images.append({
                    'image': img,
                    'classe': int(file.split('-')[0])
                })
    np.random.shuffle(images)
    train_size = int(len(images) * 0.85)
    train_images = images[:train_size]
    test_images = images[train_size:]
    write_file(train_images, "./bin/beans_train.dat")
    write_file(test_images, "./bin/beans_test.dat")

def write_file(images, filepath):
    print(f"Escrevendo arquivo: {filepath}")
    features = []
    labels = []
    for img_data in images:
        img = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2GRAY)
        classe = img_data['classe']
        
        feats = extract_features(img)

        features.append(feats)
        labels.append(classe)

    with open(filepath, 'wb') as f:
        for feature, label in zip(features, labels):
            f.write(f"{feature}   {label}\n".encode('utf-8'))

def extract_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = np.sum(hist)
    media = np.sum(hist * np.arange(256)) / total_pixels
    desvioPadrao = np.sqrt(np.sum(hist * (np.arange(256) - media) ** 2) / total_pixels)
    cumsum = np.cumsum(hist)
    mediana = np.searchsorted(cumsum, total_pixels // 2)
    return f"  {mediana:4d}   {desvioPadrao:.10e}"

def buildMLPerceptron(treinamento, teste, treinamento_classes, teste_classes):
    if os.path.exists('./out/classifier.joblib'):
        os.remove('./out/classifier.joblib')
    classificador = MLPClassifier(
        hidden_layer_sizes=250, max_iter=1000,
        activation='relu', 
        solver='adam', 
        verbose=1, 
        random_state=None, 
        learning_rate='adaptive')
    classificador.fit(treinamento, treinamento_classes)

    dump(classificador, './out/classifier.joblib')
    tentativas = classificador.predict(teste)
    score = metrics.accuracy_score(teste_classes, tentativas)
    print(f"Acurácia: {score*100:.2f}%")
    print(f"Relatório de Classificação:\n{metrics.classification_report(teste_classes, tentativas)}")

main()

# 47 iterações, 0.652778, 150 neurônios, adam, relu
# 27 iterações, 0.430556, 100 neurônios, adam, relu
# 43 iterações, 0.569444, 300 neurônios, adam, relu
# 45 iterações, 0.722222, 250 neurônios, adam, relu
# 38 iterações, 0.694444, 200 neurônios, adam, relu
# 100 iterações, 0.402778, 250 neurônios, sgd, relu
# 67 iterações, 0.555556, 350 neurônios, sgd, identity