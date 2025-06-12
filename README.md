# 🌿 GarDoc – AI-Powered Plant Diagnosis System

---

### 🖼️ Screenshots
![Upload Interface](images/interface1.jpg)

![Upload Interface](images/interface2.jpg)

---

## 📖 Project Overview  
**GarDoc** is an intelligent web-based application designed to identify garden plants and diagnose their health conditions through image classification. Built using a **deep learning model (ResNet50)**, GarDoc can detect whether a plant is healthy or unhealthy and provides care suggestions for affected plants.

The system is trained on a custom dataset of commonly found vegetable and ornamental plants. It offers a practical solution for gardeners, farmers, and hobbyists to ensure plant well-being using AI.

---

## ✅ Key Features

### 🌱 Plant Identification  
- Classifies plant species from uploaded leaf or flower images.  
- Trained on 8 distinct plant categories, each with healthy and unhealthy examples.

### 🚨 Health Diagnosis  
- Detects whether the plant is **healthy** or **unhealthy** based on visual symptoms.  
- Provides intelligent care suggestions for unhealthy plants.

### 🧠 Deep Learning-Based Classifier  
- Uses **ResNet50**, a powerful convolutional neural network, for accurate image classification.  
- Handles both **in-class classification** and **out-of-class detection** using an 'unknown' category.

### 📷 Image Upload and Processing  
- Accepts user-uploaded images in common formats (JPG, PNG).  
- Automatically resizes and preprocesses input for model prediction.

### 🧾 Care Suggestion Module  
- Displays plant-specific remedies, watering advice, and soil suggestions when an unhealthy plant is detected.

---

## 🛠 Technical Stack

- **Frontend**: HTML, CSS, JavaScript (for user interaction and image upload)
- **Backend**: Python (Flask or Streamlit)
- **AI Model**: ResNet50 (TensorFlow/Keras)
- **Dataset**:
  - 8 plant types (e.g., tomato, brinjal, lemon) with healthy and diseased variants

---

## 🎯 Objectives Achieved

- Built a complete **image classification pipeline** using deep learning.  
- Applied **transfer learning** on ResNet50 to achieve high model accuracy.  
- Developed a **user-friendly web interface** to make AI accessible to non-technical users.  
- Created a lightweight, deployable solution that bridges technology and agriculture.  
- Demonstrated **end-to-end AI integration** from data preprocessing to actionable output.
