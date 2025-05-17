from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    WORKDIR: str = os.path.dirname(os.path.dirname(__file__))
    DATADIR: str = os.path.join(WORKDIR, "data") 
    YOLO_DATASET_PATH: str = os.path.join(DATADIR, "YOLO_sdr_data")
    COCO_DATASET_PATH: str = os.path.join(DATADIR, "train")
    