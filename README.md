# Disease-detection-from-Chest-X-Ray


This project implements a **deep learning-based system** for detecting diseases from chest X-ray images, using patient history and image classification techniques. The models leverage past medical history and X-ray imaging to assist in early diagnosis.

---

## **Project Overview**

Chest X-rays are a key diagnostic tool for identifying pulmonary conditions such as pneumonia, tuberculosis, and other respiratory diseases. Automating detection using **deep learning** improves efficiency, reduces human error, and assists medical professionals in decision-making.

This project combines:
- **Chest X-ray image classification** using deep learning
- **Data augmentation** to enhance model robustness

---

## **Models Used**

The project experiments with three approaches:

1. **CNN (Convolutional Neural Network)**  
   - Standard CNN trained from scratch on chest X-ray images.
   - Used for baseline performance.

2. **CNN with Pre-trained Weights**  
   - CNN initialized with weights from a pre-trained model (e.g., ImageNet).  
   - Fine-tuned on the X-ray dataset for faster convergence and improved accuracy.

3. **MobileNetV2**  
   - Lightweight, efficient CNN architecture optimized for mobile and embedded applications.
   - Pre-trained on ImageNet, then fine-tuned on chest X-ray data.

---

## **Dataset**

The project uses a publicly available chest X-ray dataset (e.g., [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)):

- **Train**: Labeled X-ray images for model training.
- **Test**: For evaluating model performance.
- **Validation**: Optional, for hyperparameter tuning.

**Note:** Ensure patient data privacy when using real medical records.

---

## **Requirements**

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Google Colab (optional) or local GPU-enabled environment

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
