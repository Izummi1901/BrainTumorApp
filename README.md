# Brain Tumor MRI Classifier

A deep learning-based web application that detects and classifies brain tumors from MRI scans using a Convolutional Neural Network (CNN). The system predicts whether the MRI image contains glioma, meningioma, pituitary tumor, or no tumor through an interactive Streamlit interface.

# Features

- Upload MRI images in JPG/PNG format
- CNN-based multiclass tumor classification
- Real-time prediction with confidence score
- Probability distribution visualization
- Interactive and user-friendly Streamlit dashboard
- Detailed prediction analysis using charts and tables
  
# Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Pillow (PIL)

# Model Details

- CNN architecture for MRI classification
- Input image size: 128×128 RGB
- Classes:
Glioma
Meningioma
Pituitary Tumor
No Tumor

# Installation

git clone https://github.com/your-username/Brain-Tumor-MRI-Classifier.git
cd Brain-Tumor-MRI-Classifier

# Install Dependencies

pip install streamlit tensorflow numpy pandas pillow

# Run the Application

streamlit run app.py

# Project Structure

- app.py
- best_cnn_multiclass.h5
- README.md

# Disclaimer

This project is developed for educational and research purposes only and should not be used as a standalone medical diagnostic tool.

# Future Improvements

- Grad-CAM visualization
- Better CNN architectures
- Cloud deployment
- 3D MRI support
- Explainable AI integration
