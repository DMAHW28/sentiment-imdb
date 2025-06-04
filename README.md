# sentiment-imdb
This project implements a sentiment analysis model using a Transformer encoder architecture, trained on the IMDb movie reviews dataset. The goal is to classify reviews as **positive** or **negative**.

## 🧠 Model Overview

- **Architecture**: Custom Transformer encoder
- **Framework**: PyTorch
- **Dataset**: IMDb Large Movie Review Dataset
- **Task**: Binary sentiment classification (positive / negative)

## 📊 Results

- **Accuracy**: 89%
- **F1 Score**: 89%

## 🧪 Notebooks

- 📘 `train_model.ipynb`  
  Contains the full model training pipeline, including data preprocessing, model definition, and training loop.

- 📗 `evaluate.ipynb`  
  Loads the trained model and evaluates it on the test set. Also displays accuracy, F1 score, and some prediction examples.

To run the notebooks:

```bash
jupyter notebook train_model.ipynb
jupyter notebook evaluate.ipynb
