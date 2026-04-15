from inference_sdk import InferenceHTTPClient
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ==========================
# Init Roboflow Client
# ==========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="gx9YwBQt8OaWxTofW0Fb"
)

MODEL_ID = "trash-detection_img-classification/17"

# ==========================
# Folder validation images
# ==========================
VALID_FOLDER = r"D:\Belajar Data anotator\trash_detection\dataset_trash_detection\valid"

true_labels = []
pred_labels = []

# ==========================
# Loop semua gambar
# ==========================
for class_name in os.listdir(VALID_FOLDER):
    class_path = os.path.join(VALID_FOLDER, class_name)

    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_path, img)

                try:
                    result = CLIENT.infer(img_path, model_id=MODEL_ID)
                    prediction = result["top"]

                    # Samakan format label
                    true_labels.append(class_name.lower())
                    pred_labels.append(prediction.lower())

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

print("Total images evaluated:", len(true_labels))
print("Unique True Labels:", set(true_labels))
print("Unique Pred Labels:", set(pred_labels))
# ==========================
# Confusion Matrix
# ==========================
labels = sorted(list(set(true_labels)))

cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# ==========================
# Classification Report
# ==========================
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels))
print("Macro F1:", f1_score(true_labels, pred_labels, average="macro"))
print("Weighted F1:", f1_score(true_labels, pred_labels, average="weighted"))
