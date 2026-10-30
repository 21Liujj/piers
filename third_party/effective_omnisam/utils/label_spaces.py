CITYSCAPES19_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES13_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "car",
]


def get_classes(dataset_name: str, class_set: int):
    """
    Return class names for supported dataset/class_set.

    Notes:
    - class_set=19 is only supported for Cityscapes-like TrainIds space.
    - class_set=13 follows the repo's legacy mapping used for UDA experiments.
    """
    if class_set == 13:
        return CITYSCAPES13_CLASSES
    if class_set == 19:
        return CITYSCAPES19_CLASSES
    raise ValueError(f"Unsupported class_set={class_set}")

