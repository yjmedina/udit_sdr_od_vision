from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# Train the model with MPS
results = model.train(data="self_driving_yolo/dataset.yaml",  device="mps", cfg="ul/default.yml")