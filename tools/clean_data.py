import os
from tqdm import tqdm
import json


def del_augmented_images(image_folder, annotation_folder):
    image_list = [_ for _ in os.listdir(image_folder) if _.endswith(".jpg")]
    for image_name in tqdm(image_list):
        annotation_name = image_name.replace(".jpg", ".json")
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, annotation_name)
        # Remove images without annotations
        if not os.path.exists(annotation_path):
            os.remove(image_path)
            continue
        # Remove augmented images (flip/rotate) and their annotations
        if "flip" in image_name or "rotate" in image_name:
            os.remove(image_path)

            os.remove(annotation_path)


def del_duplicated_images(target_image_folder, image_folder, annotation_folder):
    target_image_list = [_ for _ in os.listdir(target_image_folder) if _.endswith(".jpg")]
    image_list = [_ for _ in os.listdir(image_folder) if _.endswith(".jpg")]
    for image_name in tqdm(image_list):
        annotation_name = image_name.replace(".jpg", ".json")
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, annotation_name)
        # Duplicate found
        if image_name in target_image_list:
            os.remove(image_path)
            os.remove(annotation_path)


def del_annotation_without_image(image_folder, annotation_folder):
    # Remove annotation files without corresponding images
    annotation_list = [_ for _ in os.listdir(annotation_folder) if _.endswith(".json")]
    for annotation in tqdm(annotation_list):
        annotation_path = os.path.join(annotation_folder, annotation)
        with open(annotation_path, "r") as f:
            anno = json.load(f)
        image_name = anno["imagePath"]
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            os.remove(annotation_path)


def get_defect_types(anno: dict):
    labels_to_names = {
        "crack_horizontal": "horizontal crack",
        "crack_vertical": "vertical crack",
        "crack_web": "network crack",
        "crack_block": "block crack",
        "crack_concrete": "concrete wall crack",
        "crack_linear": "linear/irregular crack",
        "crack_edge": "edge crack",
        "patch": "patch",
        "spalling": "spalling",
        "exposed_bars": "exposed bars",
        "pothole_dry": "dry pothole",
        "pothole_watering": "water pothole",
        "water_leakage": "water leakage",
        "efflorescence": "efflorescence",
    }
    shapes = anno["shapes"]
    defect_types = set()
    for shape in shapes:
        defect_types.add(labels_to_names[shape["label"]])
    return list(defect_types)


def del_no_target_defect_annotation(image_folder, annotation_folder):
    target_defects = ["horizontal crack", "vertical crack", "network crack", "block crack", "edge crack", "concrete wall crack", "dry pothole", "water pothole"]
    annotation_list = [_ for _ in os.listdir(annotation_folder) if _.endswith(".json")]
    for annotation in tqdm(annotation_list):
        annotation_path = os.path.join(annotation_folder, annotation)
        with open(annotation_path, "r") as f:
            anno = json.load(f)
        image_name = anno["imagePath"]
        defect_types = get_defect_types(anno)
        # Remove annotation and image if any defect type is not in target_defects
        for defect_type in defect_types:
            if defect_type not in target_defects:
                os.remove(annotation_path)
                image_path = os.path.join(image_folder, image_name)
                os.remove(image_path)
                break


if __name__ == "__main__":
    image_folder = "data2/JPEGImages_multiscale"
    annotation_folder = "data2/specific_multiscale"
    # del_augmented_images(image_folder, annotation_folder)
    # del_duplicated_images("data1/JPEGImages_multiscale", image_folder, annotation_folder)
    del_annotation_without_image(image_folder, annotation_folder)
    # del_no_target_defect_annotation(image_folder, annotation_folder)
