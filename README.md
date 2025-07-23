
#  Brain Tumor MRI Image Classification

This project applies deep learning to classify brain tumors from MRI images into four categories using both a custom CNN and transfer learning with ResNet50. A Streamlit web application is included for real-time tumor detection from uploaded images.

---

##  Project Overview

- **Domain**: Medical Imaging  
- **Goal**: Classify brain tumors into 4 types from MRI images:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

- Two deep learning approaches used:
  -  Custom Convolutional Neural Network
  -  Transfer Learning (ResNet50)

---

##  Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- NumPy / Pandas / Matplotlib / Seaborn
- OpenCV / PIL

---

##  Workflow

1. **Understand Dataset**
   - Visualize sample images
   - Check class imbalance

2. **Preprocessing**
   - Resize images to 224x224
   - Normalize pixel values

3. **Augmentation**
   - Rotation, flip, zoom, brightness shift

4. **Modeling**
   - Custom CNN
   - Transfer Learning (ResNet50)

5. **Training**
   - Callbacks: EarlyStopping, ModelCheckpoint

6. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Bar chart comparison

7. **Deployment**
   - Streamlit app for real-time classification




