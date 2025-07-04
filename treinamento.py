import os
import cv2
import time
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from joblib import dump
from utils import reveal_edges, prepare_image, extract_features

IMAGES = [
    {"path": "./data/bom_1.jpg", "classe": 1},
    {"path": "./data/bom_2.jpg", "classe": 1},
    {"path": "./data/bom_3.jpg", "classe": 1},
    {"path": "./data/bom_4.jpg", "classe": 1},
    {"path": "./data/ruim_1.jpg", "classe": 0},
    {"path": "./data/ruim_2.jpg", "classe": 0},
    {"path": "./data/ruim_3.jpg", "classe": 0},
    {"path": "./data/ruim_4.jpg", "classe": 0}
]

# Classe 1 - Bom, 0 - Ruim
def main():
    cleanup()
    #if not os.path.exists("./out"):
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
    if os.path.exists("./bin"):
        for file in os.listdir("./bin"):
            os.remove(os.path.join("./bin", file))
        os.rmdir("./bin")
    
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
    edges = reveal_edges(blurred)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 650:
            x, y, w, h = cv2.boundingRect(contour)
            if 1 <= w / h <= 4:
                bean_cropped = enhanced[y:y+h, x:x+w]

                bean_no_bg = prepare_image(bean_cropped)

                output_path = f"./out/{classe}-bean-{id*i}.jpg"
                cv2.imwrite(output_path, bean_no_bg)

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
        
        (mediana, 
         desvioPadrao, 
         entropia,
         razao_largura_altura,
         circularidade) = extract_features(img)
        feats = f"  {mediana:.10e}   {desvioPadrao:.10e}   {entropia:.10e}   {razao_largura_altura:.10e}   {circularidade:.10e}"

        features.append(feats)
        labels.append(classe)

    with open(filepath, 'wb') as f:
        for feature, label in zip(features, labels):
            f.write(f"{feature}   {label}\n".encode('utf-8'))

def buildMLPerceptron(treinamento, teste, treinamento_classes, teste_classes):
    if os.path.exists('./out/classifier.joblib'):
        os.remove('./out/classifier.joblib')
    classificador = MLPClassifier(
        hidden_layer_sizes=350, max_iter=850,
        activation='relu', 
        solver='sgd', 
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