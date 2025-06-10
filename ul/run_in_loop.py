from ultralytics import YOLO, settings
import os
import yaml

PATH_TO_DATASET = "data\YOLO_sdr_data\dataset.yaml"

DEFAULT_MODEL = "yolo11n.pt"

def main(
        confpath: str,
        path_to_dataset: str = PATH_TO_DATASET,
):
    abs_path = os.path.abspath(path_to_dataset)
    assert os.path.exists(path_to_dataset)
    
    with open(confpath) as f:
        config: dict = yaml.safe_load(f)
    model_name = config['model'] or DEFAULT_MODEL
    print(model_name)
    model = YOLO(model_name)  # load a pretrained model (recommended for training)
    _ = model.train(data=abs_path, cfg=confpath, device=0)

if __name__ == '__main__':
    import mlflow
    workdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    TRACKING_URI = f"file:///{workdir}/mlruns"
    mlflow.set_tracking_uri(TRACKING_URI)
    settings.update(
        {"tensorboard": True, "mlflow": True}
        )

    CONFS = [
        # "ul/adamW_e50.yaml",
        # "ul/basicAugmentation_E50.yaml",
        "ul/basicAugmentation_adamW_E50.yaml",
        "ul/yolo11m_basicAugmentation_adamW_E50.yaml",
    ]

    for conf in CONFS:
        print(conf)
        main(conf)