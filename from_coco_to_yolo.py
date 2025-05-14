import os
import json
from collections import defaultdict
from util import process_annotation
import shutil
import yaml
from typing import List, Dict, Any, Literal, Tuple
from tqdm.auto import tqdm
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

Image = Dict[str, str]
Annotation = Dict[str, str]


def split_images(
    images: List[Image],
    id_to_annotations: Dict[str, List[Annotation]],
    n_classes: int,
    valid_size: float) -> Tuple[List[Image], List[Image]]:

    labels = np.zeros((len(images), n_classes), dtype=np.int32)
    for i, image in enumerate(images):
        anns = id_to_annotations[image['id']]
        for ann in anns:
            labels[i, ann['category_id'] - 1] = 1

    indices = np.arange(len(images))[:, np.newaxis]

    train_indices, train_labels, valid_indices, valid_labels = iterative_train_test_split(
        indices,
        labels,
        test_size=valid_size)

    print(f"train distribution: {train_labels.mean(axis=0)}")
    print(f"valid distribution: {valid_labels.mean(axis=0)}")
    train_images = [images[i] for i in train_indices.ravel()]
    valid_images = [images[i] for i in valid_indices.ravel()]
    return train_images, valid_images


def write_images(
        origin_images_dir: str,
        path: str, 
        images: List[Dict[str, Any]],
        id_to_annotations: Dict[str, List[Dict[str, Any]]],
        set_type: Literal['train', 'val']) -> None:

    images_dir = os.path.join(path, "images", set_type)
    labels_dir = os.path.join(path, "labels", set_type)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for image in tqdm(images):
        annotations = id_to_annotations[image['id']]
        yolo_annotations = [process_annotation(ann, image) for ann in annotations]
        yolo_annotations_str = "\n".join(yolo_annotations)

        output_image_path = os.path.join(images_dir, image['file_name'])
        output_labels_path = os.path.join(labels_dir, image['file_name'].replace(".jpg", ".txt"))
        
        with open(output_labels_path, 'w') as f:
            f.write(yolo_annotations_str)
            
        input_image_path = os.path.join(origin_images_dir, image['file_name'])
        shutil.copy(input_image_path, output_image_path)


def from_coco_to_yolo(
    coco_train_dir: str,
    yolo_output_dir: str,
    valid_size: float):
    coco_labels_json = os.path.join(coco_train_dir, "labels.json")
    assert os.path.exists(coco_labels_json)

    with open(coco_labels_json) as f:
        coco_labels = json.load(f)
    
    categories = {
        i: category['name']
        for i, category in enumerate(coco_labels['categories'][1: ])
    }

    yolo_config = {
        # dataset root dir (absolute or relative; if relative, it's relative to default datasets_dir)
        # "path": yolo_output_dir, 
        "train": "images/train", # train images (relative to 'path') 4 images
        "val": "images/val", # val images (relative to 'path') 4 images
        "names": categories
    }

    os.makedirs(yolo_output_dir, exist_ok=True)
    with open(os.path.join(yolo_output_dir, "dataset.yaml"), 'w') as f:
        yaml.dump(yolo_config, f)

    id_to_annotations = defaultdict(list)
    for ann in coco_labels['annotations']:
        id_to_annotations[ann['image_id']].append(ann)


    images = coco_labels['images']

    train_images, valid_images = split_images(
        images,
        id_to_annotations,
        n_classes=len(categories),
        valid_size=valid_size,
    )

    origin_images_dir = os.path.join(coco_train_dir, "data")


    print(origin_images_dir)

    write_images(origin_images_dir, yolo_output_dir, train_images, id_to_annotations, set_type='train')
    write_images(origin_images_dir, yolo_output_dir, valid_images, id_to_annotations, set_type='val')


if __name__ == '__main__':
    from_coco_to_yolo("train", "self_driving_yolo", valid_size=0.2)

