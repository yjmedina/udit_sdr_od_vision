from ultralytics import YOLO
import os

PATH_TO_DATASET = "data\YOLO_sdr_data\dataset.yaml"

if __name__ == '__main__':
    path_to_dataset = os.path.abspath(PATH_TO_DATASET)
    assert os.path.exists(path_to_dataset)
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # Train the model with MPS
    results = model.train(data=path_to_dataset, cfg="ul/default.yml", device=0)