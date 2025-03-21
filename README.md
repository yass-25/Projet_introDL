# Projet_introDL

# BiLSTM-CRF for Named Entity Recognition (NER)

Ce projet vise à reproduire les résultats de l’article **"Bidirectional LSTM-CRF Models for Sequence Tagging"** (Huang et al., 2015), en appliquant une architecture BiLSTM-CRF à la tâche de reconnaissance d’entités nommées (NER) sur le dataset **CoNLL-2003**.

## 🎯 Objectif

- Implémenter un modèle BiLSTM-CRF en PyTorch.
- Entraîner ce modèle sur les données CoNLL-2003.
- Suivre les performances via TensorBoard.
- Appliquer des techniques modernes comme l’early stopping, le learning rate scheduler, la régularisation (dropout, weight decay).
- Observer le comportement du modèle : overfitting, stabilité des gradients, etc.

## 📦 Structure du projet

├── main.ipynb # Notebook principal avec tout le pipeline (prétraitement, modèle, entraînement) ├── model.py # Définition du modèle BiLSTM-CRF (optionnel si séparé) ├── requirements.txt # Librairies nécessaires ├── runs/ # Logs TensorBoard ├── README.md # Ce fichier


## 📚 Dataset

Le dataset **CoNLL-2003** est automatiquement téléchargé via la librairie 🤗 `datasets`.

```python
from datasets import load_dataset
dataset = load_dataset("conll2003")
⚙️ Prérequis

Ce projet est conçu pour être exécuté sur Google Colab avec GPU activé.

Installation des dépendances (Colab)
!pip install datasets transformers torch torchvision torchaudio torchcrf tensorboard
🏗️ Entraînement du modèle

L'entraînement se fait via une fonction train_model() qui :

applique un early stopping,
ajuste dynamiquement le learning rate avec un scheduler,
enregistre les courbes dans TensorBoard,
log les gradients et poids pour surveiller les problèmes de vanishing/exploding gradients.
Lancer l'entraînement :
train_model(model, train_dataloader, val_dataloader, epochs=20, learning_rate=0.0005)
📊 Visualisation avec TensorBoard

Dans Colab :

%load_ext tensorboard
%tensorboard --logdir=runs/
Vous y verrez :

Loss/Train
Loss/Validation
Accuracy
Learning Rate
Histograms des poids et gradients
🔍 Résultats obtenus

Paramètres	Score
hidden_dim = 256	
dropout = 0.5	
learning rate = 0.001	
loss initiale	2.19
perte finale (val)	~1.6
Malgré des performances encourageantes, le modèle n’atteint pas les scores exacts de l’article original, probablement à cause de différences dans le tokenizer, l’optimisation, et le fine-tuning non réalisé sur BERT.
✏️ Auteur

Projet réalisé dans le cadre d'un travail académique en Deep Learning.
