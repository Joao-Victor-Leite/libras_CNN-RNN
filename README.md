# Reconhecimento do Alfabeto de LIBRAS com MediaPipe e LSTM

![Licença](https://img.shields.io/badge/licença-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)

> Este projeto é referente ao meu Trabalho de Conclusão de Curso para Ciência da Computação na Universidade Estadual de Santa Cruz (UESC). Ele implementa um sistema em tempo real para o reconhecimento de letras do alfabeto da Língua Brasileira de Sinais (LIBRAS). A solução utiliza a biblioteca **MediaPipe** para a extração de pontos-chave (keypoints) das mãos e uma **Rede Neural Recorrente (LSTM)** para classificar as sequências de sinais capturadas por uma webcam.

---

## (Demonstração)

![Demonstração do Projeto](video_demonstracao.gif)

---

## 📖 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Começando](#-começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Uso](#-uso)
  - [1. Coleta de Dados](#1-coleta-de-dados)
  - [2. Treinamento do Modelo](#2-treinamento-do-modelo)
  - [3. Inferência em Tempo Real](#3-inferência-em-tempo-real)
- [Licença](#-licença)

---

## 💻 Sobre o Projeto

O objetivo principal é criar uma ferramenta acessível que possa traduzir os sinais estáticos e dinâmicos do alfabeto de LIBRAS em texto. O fluxo de trabalho do projeto é o seguinte:

1.  **Captura de Vídeo:** Uma webcam captura o vídeo do usuário fazendo os sinais.
2.  **Extração de Pontos-Chave:** O MediaPipe Holistic processa cada frame para detectar e extrair as coordenadas dos pontos das mãos.
3.  **Processamento de Sequências:** Os pontos-chave de 30 frames consecutivos são agrupados para formar uma sequência, que representa um sinal completo.
4.  **Treinamento do Modelo:** Um modelo LSTM é treinado para aprender a classificar cada letra do alfabeto.
5.  **Reconhecimento em Tempo Real:** O script de inferência utiliza o modelo treinado para prever a letra que está sendo sinalizada em tempo real.

---

## ✨ Tecnologias Utilizadas

As seguintes ferramentas e bibliotecas foram essenciais para a construção deste projeto:

- **[Python 3.9+](https://www.python.org/)**: Linguagem de programação principal.
- **[TensorFlow](https://www.tensorflow.org/)**: Framework para criação e treinamento do modelo de Deep Learning.
- **[OpenCV](https://opencv.org/)**: Para captura e processamento de imagem da webcam.
- **[MediaPipe](https://mediapipe.dev/)**: Para detecção e rastreamento de pontos-chave das mãos em tempo real.
- **[Scikit-learn](https://scikit-learn.org/)**: Para avaliação de métricas do modelo, como a matriz de confusão.
- **[NumPy](https://numpy.org/)**: Para manipulação eficiente de arrays e dados numéricos.
- **[Jupyter Notebook](https://jupyter.org/)**: Para o ambiente de treinamento do modelo.

---

## 📂 Estrutura do Projeto

```
libras_CNN-RNN/
├── main/
│   ├── lstm_inference.py     # Script para inferência em tempo real com a webcam
│   └── train_lstm.ipynb      # Notebook para treinamento do modelo LSTM
├── hand_capture/
│   ├── utils.py              # Funções utilitárias (extração de keypoints, etc.)
│   └── video_collector.py    # Script para coletar os dados de vídeo para treinamento
├── models/                   # Diretório para salvar os modelos treinados (.h5)
├── dataset/
│   └── lstm/
│       ├── train/            # Dados de treino (gerados pelo video_collector.py)
│       └── test/             # Dados de teste (gerados pelo video_collector.py)
└── README.md                 # Este arquivo
```

---

## 🚀 Começando

Siga os passos abaixo para ter uma cópia local do projeto funcionando.

### Pré-requisitos

Você precisa ter o Python 3.9 (ou superior) e o pip instalados. Recomenda-se o uso de um ambiente virtual.

```sh
# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### Instalação

1.  Clone o repositório:
    ```sh
    git clone https://github.com/Joao-Victor-Leite/libras_CNN-RNN.git
    cd libras_CNN-RNN
    ```

2.  Instale as dependências a partir do arquivo `requirements.txt` (crie um se não houver):
    ```sh
    pip install opencv-python mediapipe tensorflow numpy scikit-learn matplotlib seaborn jupyter
    ```

---

## 🏃 Uso

O projeto é dividido em três etapas principais:

### 1. Coleta de Dados

Para treinar um novo modelo, você precisa primeiro coletar os dados. Execute o script `video_collector.py`. Ele irá guiar você na gravação de vídeos curtos para cada letra do alfabeto, salvando os pontos-chave extraídos na estrutura de pastas correta dentro de `dataset/lstm/`.

```sh
python hand_capture/video_collector.py
```

### 2. Treinamento do Modelo

Com os dados coletados, você pode treinar o modelo LSTM. Abra e execute o notebook `train_lstm.ipynb` usando o Jupyter ou Google Colab, por utilizar o poder de processamento dos servidores do Google.

```sh
jupyter notebook main/train_lstm.ipynb
```

O notebook irá processar os dados, construir a arquitetura do modelo, treiná-lo e salvar o arquivo `.h5` final no diretório `models/`.

### 3. Inferência em Tempo Real

Após treinar e salvar um modelo, execute o script `lstm_inference.py` para iniciar o reconhecimento em tempo real. O script listará os modelos disponíveis na pasta `models/` e pedirá que você escolha qual carregar.

```sh
python main/lstm_inference.py
```

---

## 📝 Licença

Distribuído sob a Licença MIT. Veja `LICENSE` para mais informações.

---
