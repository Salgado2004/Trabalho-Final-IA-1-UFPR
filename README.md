Tecnologia em Análise e Desenvolvimento de Sistemas

Setor de Educação Profissional e Tecnológica - SEPT

Universidade Federal do Paraná - UFPR

---

*DS803 - Inteligência Artificial Aplicada I*

Prof. Roberto Tadeu Raittz

# Classificação de Feijões bons e ruins com MLP

## Descrição do projeto

O projeto consiste em classificar feijões bons e ruins utilizando uma rede neural MLP (Multi-Layer Perceptron). O dataset utilizado contém imagens de feijões, e o objetivo é treinar um modelo que possa identificar se um feijão é bom ou ruim com base em suas características visuais.

## Resultados

Os resultados obtidos dos testes estão salvos na pasta `snapshots/`, onde as imagens dos feijões estão contornadas com as classificações obtidas pelo modelo. Os feijões considerados "bons" são marcados com um contorno verde, enquanto os "ruins" são marcados com um contorno vermelho.

## Como executar o projeto
1. Certifique-se de ter o Python 3.x instalado em seu sistema.
2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script de treinamento:
   ```bash
    python treinamento.py
    ```
4. Após o treinamento, você pode testar o modelo as imagens de feijões utilizando o script de teste:
    ```bash
    python teste.py
    ```

## Alunos
- Gabriel Troni [@Gabriel-Troni](https://github.com/Gabriel-Troni)
- Leonardo Felipe Salgado [@Salgado2004](https://github.com/Salgado2004)
- Raul Ferreira Bana [@RaulBana](https://github.com/RaulBana)