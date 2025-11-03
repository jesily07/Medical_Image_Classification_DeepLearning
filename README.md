# COVID-19 Chest X-Ray Deep Learning Classification  
> **Detecting COVID-19 from chest X-ray images using a fine-tuned ResNet50 CNN model**  
> **Built with TensorFlow | Streamlit | Grad-CAM Explainability**

## 1. Project Overview

This project demonstrates how deep learning can be applied in the healthcare domain to assist in **diagnosing COVID-19 from chest X-rays**.  
The model leverages **transfer learning (ResNet50)** to achieve high accuracy with limited data, and includes **explainable AI (Grad-CAM)** for visual interpretability.  
A simple **Streamlit UI** enables real-time image upload, prediction, and heatmap visualization.

---

## 2. Business & Problem Understanding

- **Objective:** Automate early COVID-19 screening using X-ray images.  
- **Business Impact:** Quick triage assistance for radiologists, reducing diagnostic turnaround time.  
- **Solution Overview:**  
  1. Preprocess medical images.  
  2. Fine-tune a pre-trained CNN.  
  3. Visualize model explanations interactively.

---

## 3. Project Architecture

Data Source → Data Preprocessing → Model Training → Evaluation → Grad-CAM → Streamlit UI

| Step | Component                 | Description                                      |
|------|---------------------------|--------------------------------------------------|
| 1    | **Data Preprocessing**    | Cleaning, resizing, label encoding               |
| 2    | **Model Building**        | Transfer learning using ResNet50                 |
| 3    | **Training & Validation** | Fine-tuning with early stopping and augmentation |
| 4    | **Evaluation**            | Accuracy, confusion matrix, ROC                  |
| 5    | **Explainability**        | Grad-CAM visualization                           |
| 6    | **Deployment**            | Streamlit app for local inference                |

---

## 4. Data Pipeline Summary

- **Dataset:**   COVID-19 Radiography Database (Kaggle)  
- **Classes:**   `COVID-19`, `Normal`, `Pneumonia`  
- **Images:**    ~15,000  
- **Preprocessing:**
  - Resize → 224×224
  - Normalize → [0, 1]
  - Data Augmentation → rotation, zoom, horizontal flip

---

## 5. Model Development (ResNet50 Fine-Tuning)

- **Base Model:** `ResNet50` pretrained on ImageNet  
- **Fine-tuning Strategy:**
  ```python
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
  for layer in base_model.layers[:100]:
      layer.trainable = False
  model = Sequential([
      base_model,
      GlobalAveragePooling2D(),
      Dense(512, activation='relu'),
      Dropout(0.4),
      Dense(3, activation='softmax')
  ])

Optimizer: Adam (lr = 1e-4)
Loss Function: Categorical Crossentropy
Metrics: Accuracy, Precision, Recall

---

## 6. Evaluation Results

| Metric    | Validation Score |
| --------- | ---------------- |
| Accuracy  |   97.5%**        |
| Precision |   96.8%          |
| Recall    |   97.2%          |
| F1-Score  |   97.0%          |

Visualizations:
- Confusion matrix
- ROC curve
- Grad-CAM heatmap overlays

---

## 7. Explainability (Grad-CAM)

- Applied Grad-CAM on the final convolutional layer to highlight important regions.
- Visual heatmaps clearly show the model focuses on lung regions for COVID-positive predictions.
- Adds interpretability and builds clinical trust.

---

## 8. Streamlit Application

A lightweight Streamlit dashboard demonstrates the model’s predictions in real time.

Features:
1. Upload X-ray image
2. Predict class and probability
3. View Grad-CAM heatmap

Run locally:
streamlit run streamlit_app.py

---

## 9. Local Setup Instructions

1. Clone the repository
git clone https://github.com/jesily07/Projects.git
cd Projects/COVID-19_Chest_X-Ray_DeepLearning_Classification

2. Create and activate environment
conda create -n covid19_dl python=3.10
conda activate covid19_dl

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run streamlit_app.py

Note: 
- The fine-tuned model (final_resnet50_finetuned.h5) is excluded from the repo due to size (>100 MB).
- For portfolio use, the model can be downloaded manually if needed.

---