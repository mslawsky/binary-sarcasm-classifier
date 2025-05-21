# Sarcasm Detection in News Headlines 📰

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-blue.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.6+-green.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue.svg)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue.svg)](https://jupyter.org/)
[![TensorFlow Embedding Projector](https://img.shields.io/badge/Embedding%20Projector-TensorBoard-blueviolet.svg)](https://projector.tensorflow.org/)

## Overview 📖

This project demonstrates a **binary sarcasm classifier** for news headlines using [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). The repository contains two different model implementations:

1. **Basic Embedding with Global Average Pooling** - A simple and efficient model for baseline performance
2. **Bidirectional LSTM** - An advanced model architecture that captures sequence context in both directions

Both models process raw text headlines, convert them into numerical sequences using text vectorization, and predict whether a headline is **sarcastic** 😏 or **not sarcastic** 📰.

---

## Features 🚀

- Advanced text preprocessing with `TextVectorization` layer 🔤
- Multiple model architectures (GlobalAveragePooling and Bidirectional LSTM) 🧠
- Efficient word embeddings 🎯
- Binary classification for sarcasm detection 🏷️
- Comprehensive training and validation metrics 📈
- TF.Data pipeline optimization for efficient processing 🔄
- Hyperparameter optimization capabilities 🛠️
- Word embedding visualization exports 🌐 for the [TensorFlow Embedding Projector](https://projector.tensorflow.org/)
- Overfitting detection and mitigation strategies 🔍

---

## Dataset 📦

- **Source:** [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Description:** News headlines labeled for sarcasm detection
- **Format:** JSON file with headlines and binary labels (`0 = not sarcastic`, `1 = sarcastic`)
- **Training Split:** 20,000 samples for training, remainder for validation
- **Preprocessing:** Text standardization, tokenization, and padding to fixed length

---

## Model Architectures 🏗️

### Basic Model with Global Average Pooling

```
Input Layer (32 tokens max)
    ↓
TextVectorization (10,000 vocab)
    ↓
Embedding Layer (16 dimensions)
    ↓
GlobalAveragePooling1D
    ↓
Dense Layer (24 units, ReLU)
    ↓
Dense Layer (1 unit, Sigmoid)
    ↓
Binary Classification Output
```

### Bidirectional LSTM Model

```
Input Layer (32 tokens max)
    ↓
TextVectorization (10,000 vocab)
    ↓
Embedding Layer (16 dimensions)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dense Layer (24 units, ReLU)
    ↓
Dense Layer (1 unit, Sigmoid)
    ↓
Binary Classification Output
```

**Key Parameters:**
- Vocabulary Size: 10,000 tokens
- Max Sequence Length: 32 tokens
- Embedding Dimensions: 16
- LSTM Units: 32 (bidirectional, resulting in 64-dimensional output)
- Training Examples: 20,000
- Padding Type: 'pre'
- Truncation Type: 'post'

---

## Getting Started 🛠️

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- [Jupyter Notebook](https://jupyter.org/)

### Installation

```bash
git clone https://github.com/yourusername/sarcasm-detection
cd sarcasm-detection
pip install -r requirements.txt
```

### Usage

1. Download the sarcasm dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) or use the provided `sarcasm.json` file.
2. Open and run the desired notebook:
   - `C3_W2_Lab_2_sarcasm_classifier.ipynb` - Basic model with GlobalAveragePooling
   - `C3_W3_Lab_5_sarcasm_with_bi_LSTM.ipynb` - Advanced model with Bidirectional LSTM
3. Follow the notebook steps to:
   - Load and preprocess the data
   - Build and compile the model
   - Train with customizable parameters
   - Evaluate performance and visualize results
4. Export embedding weights for visualization in the [TensorFlow Embedding Projector](https://projector.tensorflow.org/).

---

## 📂 Code Structure

- `C3_W2_Lab_2_sarcasm_classifier.ipynb` - Basic model implementation
- `C3_W3_Lab_5_sarcasm_with_bi_LSTM.ipynb` - Bidirectional LSTM model implementation
- `sarcasm.json` - Dataset file (download separately)
- `requirements.txt` - List of dependencies
- `vecs.tsv` - Exported word vectors (generated after training)
- `meta.tsv` - Exported metadata (generated after training)

---

## Results and Performance Comparison 📊

### Basic Model (Global Average Pooling)
- **Training Accuracy:** ~96% after 10 epochs
- **Validation Accuracy:** ~84% (with some overfitting observed)
- **Model Size:** Fewer parameters, computationally efficient
- **Training Speed:** Faster training time

![Training Curve](training-curve.png)

### Bidirectional LSTM Model
- **Training Accuracy:** ~97% after 10 epochs
- **Validation Accuracy:** ~84-85% 
- **Model Size:** More parameters (174,129) but still lightweight
- **Training Speed:** Slower than the basic model but captures sequential information
- **Advantages:** Better captures word order and context in both directions

![Training Curve](training-curve-2.png)

### Training Curves
The notebooks generate visualizations showing:
- Accuracy progression over epochs
- Loss reduction during training
- Training vs. validation performance comparison

---

## TF.Data Pipeline Optimization 🔄

Both models use TensorFlow's efficient data pipeline (`tf.data.Dataset`) with optimizations:
- Data caching
- Prefetching
- Shuffling with buffer
- Batching
- Efficient sequence padding

This results in faster training times and better resource utilization.

---

## Key Insights 🔍

1. **Bidirectional LSTM** captures word order and context in both directions, potentially improving performance on sequence-sensitive tasks
2. **Global Average Pooling** model provides a simpler architecture with fewer parameters
3. **Text Vectorization** layer provides efficient preprocessing integrated into the model
4. Both models show signs of **overfitting** after several epochs
5. **Hyperparameter tuning** opportunities exist for vocabulary size, embedding dimensions, LSTM units, and dense layer architecture

---

## TensorFlow Embedding Projector 🌐

Visualize the learned word embeddings with the [TensorFlow Embedding Projector](https://projector.tensorflow.org/):

1. After training, export embedding weights and metadata to `vecs.tsv` and `meta.tsv`.
2. Upload these files to the Embedding Projector.
3. Explore word relationships and clusters in the learned embedding space.
4. Discover how the model represents sarcastic vs. non-sarcastic language patterns.

![Demo Embedding Projector](demo.gif)

---

## Future Work 🌱

- Experiment with different architectures (GRU, Transformer-based models)
- Implement regularization techniques to reduce overfitting (dropout, L2 regularization)
- Stack multiple Bidirectional LSTM layers for deeper context understanding
- Try different vocabulary sizes and embedding dimensions
- Add attention mechanisms for better context understanding
- Explore transfer learning with pre-trained embeddings (Word2Vec, GloVe)
- Multi-class classification for different types of sarcasm
- Combine LSTM features with CNN features for hybrid models

---

## Acknowledgements 🙏

Special thanks to:
- [Andrew Ng](https://www.andrewng.org/) for creating the Deep Learning AI curriculum
- [Laurence Moroney](https://twitter.com/lmoroney) for excellent instruction and developing the course materials  
- The creators of the [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- This notebook was created as part of the TensorFlow Developer Certificate program by DeepLearning.AI

---

## Contact 📫

For inquiries about this project:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© 2025 Melissa Slawsky. All Rights Reserved.
