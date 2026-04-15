from inference_sdk import InferenceHTTPClient
import tkinter as tk
from tkinter import filedialog

# =========================
# Init Roboflow Client
# =========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# =========================
# Open file dialog
# =========================
root = tk.Tk()
root.withdraw()  # hide empty Tk window

file_path = filedialog.askopenfilename(
    title="Select an image for inference",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# =========================
# Run inference
# =========================
if file_path:
    result = CLIENT.infer(
        file_path,
        model_id="trash-detection_img-classification/17"
    )

    print("\n===== INFERENCE RESULT =====")
    print("Image Path :", file_path)
    print("Prediction :", result["top"])
    print("Confidence :", round(result["confidence"] * 100, 2), "%")

else:
    print("No image selected.")
