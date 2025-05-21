from ultralytics import YOLO
import os
from typer import Typer


cli = Typer()

PATH_TO_DATASET = "data\YOLO_sdr_data\dataset.yaml"


@cli.command()
def main(
        confpath: str,
        path_to_dataset: str = PATH_TO_DATASET,
):
    abs_path = os.path.abspath(path_to_dataset)
    assert os.path.exists(path_to_dataset)
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data=abs_path, cfg=confpath, device=0)

if __name__ == '__main__':
    cli()
