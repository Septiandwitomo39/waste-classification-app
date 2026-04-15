# 🧠 Waste Image Classification System (15 Classes)

This project presents a complete computer vision pipeline for classifying waste images into 15 categories using a ResNet34-based model.  
It covers dataset preparation, model training, evaluation, and deployment through a simple Python-based application.
Waste classification plays a crucial role in improving recycling efficiency and reducing environmental impact.  
This project demonstrates how computer vision can be applied to automate waste sorting in real-world scenarios.

---

## 🚀 Key Features

- Multi-class image classification (15 categories)
- Real-time inference using Roboflow API
- Local Python application with file selection (Tkinter GUI)
- Confidence score output
- Custom evaluation script with detailed performance analysis

---
## 📁 Project Structure

├── inference_app.py # Run inference (GUI-based)
├── evaluate_model.py # Model evaluation (confusion matrix & metrics)
├── requirements.txt
└── README.md

---
## ⚙️ Installation
Clone this repository:
git clone https://github.com/septiandwitomo39/waste-classification-app.git
cd waste-classification-app

install Dependencies :
pip install -r requirements.txt

---
---
# ▶️ Usage
## 📊 Run Evaluation

python evaluate_model.py

This Script provides :
- Confusion Matrix Visualization
- Class-wise Precision
- Macro & Weighted F1 Score
---

## 🔐 API Configuration
This project uses Roboflow API for inference
Set your API key as environtment variable :
### Windows :
set ROBOFLOW_API_KEY=your_api_key_here

### Linux :
export ROBOFLOW_API_KEY=your_api_key_here

⚠️ Internet connection is required for both inference and evaluation.

---
## 🌐 Model Deployment
- Web Demo :
  https://app.roboflow.com/septiandwitomo39/trash-detection_img-classification/models/trash-detection_img-classification/17
- Local Inference :
  Provided in this repository

---
## 📊 Model Details
- Model Architecture : ResNet34
- Task : Image Classification
- Number of Classes : 15
- Dataset Size : 462 images
- Input Size : 224 × 224
- Training Platform : Roboflow (Serverless)

---
## 📈 Performance Summary
- Accuracy : 83.9%
- Macro F1 Score : 0.78
- Weighted F1 Score : 0.83

## 🔍 Key Insights
- Strong performance on visually distinct objects
- Misclassification occurs between visually similar classes (e.g., glass vs plastic bottles)
- Performance is influenced by dataset limitations and class similarity

---
---
## ⚠️ Limitations
- Requires internet connection (API-Based inference)
- Limited Dataset Size
- Sensitive to visually similar object catagories

---
## 🔮 Future Improvements
- Implement fully offline inference (no API dependency)
- Increase dataset diversity for similar classes
- Experiment with deeper architecture (ResNet50 / EfficientNet
- Develop web-based interface (Streamlit / Flask)

---
# 👤 Author
### Septian Dwitomo
Freelance Data Annotator | Computer Vision Enthusiast

Email : septiandwitomo39@gmail.com
phone : +62 895608623500

---
### ⭐ Notes
This project is part of professional portofolio demonstrating end-to-end computer vision workflow, including data preparation, model evaluation, and deployment
