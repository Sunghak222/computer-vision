---
# 🌿 Campus Plant Classification

This project implements a complete machine learning pipeline for classifying campus plant images using handcrafted features and classical machine learning models.

The system covers dataset construction, feature extraction, model training, evaluation, and a simple application for real-time prediction.

---

## 📌 Project Overview

- **Task**: Multi-class plant classification (8 classes)
- **Approach**: Handcrafted features + classical ML models
- **Models**: SVM, Random Forest (RF), XGBoost (XGB)
- **Best Model**: Random Forest + HSV features + augmentation
- **Evaluation**: Accuracy, Macro F1, Weighted F1

---

## 🗂️ Project Structure

```

ASSIGNMENT2_CODE_21097305D_HEO_SUNGHAK/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── experiments/
│   └── class_names.json
│
├── results/
│   ├── experiments/
│   ├── misclassified/
│   ├── sample_images/
│   ├── confusion_matrix.png
│   ├── experiment_history.jsonl
│   └── prediction_records.csv
│
├── src/
│   ├── app.py
│   ├── augmentation.py
│   ├── evaluate.py
│   ├── experiment_config.py
│   ├── experiment_utils.py
│   ├── feature_extraction.py
│   ├── sanity_check.py
│   ├── split_dataset.py
│   ├── train.py
│   └── tune.py
│
├── requirements.txt

````

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
````

---

## 🚀 Usage

### 1. Dataset Preparation

Due to limited memory, dataset is stored in One Drive:
[Link to Dataset](https://connectpolyu-my.sharepoint.com/:f:/r/personal/21097305d_connect_polyu_hk/Documents/data?csf=1&web=1&e=wUKlU8)

```bash
python src/split_dataset.py
python src/sanity_check.py
```

---

### 2. Train Model

```bash
python src/train.py
```

---

### 3. Evaluate Model

```bash
python src/evaluate.py
```

Outputs:

* Confusion matrix
* Prediction records
* Misclassified images

---

### 4. Run Application

```bash
streamlit run src/app.py
```

* Upload an image through the browser
* The predicted plant class will be displayed

---

## 🔍 Features

* **HOG**: shape-based feature extraction
* **LBP / GLCM**: texture-based descriptors
* **HSV histogram**: color-based representation

---

## 📊 Key Findings

* HSV features dominated performance in validation
* Texture features contributed less than expected
* Significant gap between validation and test performance
* Dataset structure strongly influenced model performance

---

## ⚠️ Limitations

* Limited instance diversity across classes
* Possible overlap of similar plant instances across splits
* Sensitivity to lighting conditions (HSV dependency)
* Difficulty in separating visually similar classes

---

## 📬 Notes

* All experiments were conducted on a fixed dataset split
* Validation set was used for model selection
* Test set was used only for final evaluation

---

```
