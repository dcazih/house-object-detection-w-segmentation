import os
from ultralytics import YOLO
import shutil
import wandb

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_YAML = "data.yaml"
EPOCHS = 100
IMG_SIZE = 640
PROJECT_NAME = "house-segmentation"
MODEL_SIZES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"] 
SAVE_DIR = "models"  # where best.pt copies go
WANDB_PROJECT = "HouseSeg"

os.makedirs(SAVE_DIR, exist_ok=True) # Create directory for saved models

# -------------------------------
# LOOP OVER MODEL SIZES (3)
# -------------------------------
for model_file in MODEL_SIZES:
    model_name = model_file.replace(".pt", "")
    run_name = f"{model_name}-run"

    print(f"\nTraining {model_name}...")

    model = YOLO(model_file)

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_MODE"] = "online"  # "offline" if testing

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name=model_name,
        save=True,
        verbose=True
    )

    # -------------------------------
    # SAVE BEST MODEL TO CENTRAL FOLDER
    # -------------------------------
    best_model_path = f"runs/segment/{model_name}/weights/best.pt"
    if os.path.exists(best_model_path):
        dst = os.path.join(SAVE_DIR, f"{model_name}_best.pt")
        shutil.copy(best_model_path, dst)
        print(f"Found and saved best model: {dst}")
    else:
        print(f"Best model not found for {model_name}")

print("\nAll models trained and saved.")
