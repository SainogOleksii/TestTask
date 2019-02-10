import numpy as np

from typing import NewType, List, Tuple

Box = NewType("Box", List[np.float16])


class BoundingBox:
    def __init__(self, original_size: Tuple[int, int], box: Box or List[float], box_type: str = "yolo") -> None:
        if box_type not in ["yolo", "coco"]:
            raise TypeError("Type of Bounding Box can be equal only 'yolo' or 'coco'.")
        self.original_size:     Tuple[int, int] = original_size
        self.box:               Box = box if box_type == "coco" else self.yolo_to_coco(self.original_size, box)

    def get_box(self) -> Box:
        """
        :return: bounding box parameters.
        """
        return self.box

    def get_size(self) -> Tuple[int, int]:
        """
        :return: size of image.
        """
        return self.original_size

    def __repr__(self):
        """
        :return: string for representation in console.
        """
        return f"{self.box} in {self.original_size}"

    @staticmethod
    def coco_to_yolo(original_size: Tuple[int, int], coco_format: List[int] or Box) -> Box:
        """
        Transform coco format to yolo format.
        :param original_size: size of image.
        :param coco_format: bounding box parameters in coco format.
        :return: bounding box parameters in yolo format.
        """
        result = np.zeros(np.array(coco_format).shape, dtype=np.float16)
        result[0] = 0.5 * (coco_format[0] + coco_format[2])
        result[1] = 0.5 * (coco_format[1] + coco_format[3])
        result[2] = coco_format[2] - coco_format[0]
        result[3] = coco_format[3] - coco_format[1]
        result[::2] /= original_size[0]
        result[1::2] /= original_size[1]
        return list_to_box(result)

    @staticmethod
    def yolo_to_coco(original_size: Tuple[int, int], yolo_format: List[float] or Box) -> Box:
        """
        Transform yolo format to coco format.
        :param original_size: size of image.
        :param yolo_format: bounding box parameters in yolo format.
        :return: bounding box parameters in coco format.
        """
        result = np.zeros(np.array(yolo_format).shape, dtype=np.float16)
        result[0] = yolo_format[0] - 0.5 * yolo_format[2]
        result[1] = yolo_format[1] - 0.5 * yolo_format[3]
        result[2] = yolo_format[0] + 0.5 * yolo_format[2]
        result[3] = yolo_format[1] + 0.5 * yolo_format[3]
        result[::2] *= original_size[0]
        result[1::2] *= original_size[1]
        return list_to_box(result, np.int16)

    @staticmethod
    def get_full_box(bbox: Box) -> np.ndarray:
        """
        Return all coordinates of bounding box.
        :param: box coordinates.
        :return: array of coordinates pairs.
        """
        result = [np.array(bbox[:2])]
        result += [np.array([bbox[2], bbox[1]]), np.array([bbox[0], bbox[3]]), np.array(bbox[-2:])]
        return np.array(result)

    @staticmethod
    def get_max_box(full_box: np.ndarray):
        """
        Create the least box that include input bounding box.
        :param full_box: all bounding box coordinates.
        :return: new box with Box type.
        """
        return list_to_box([full_box.T[0].min(), full_box.T[1].min(), full_box.T[0].max(), full_box.T[1].max()],
                           np.int16)


def list_to_box(list_: List[float] or np.ndarray, datatype: type = np.float16) -> Box or Exception:
    """
    Transform list of floats to Box type.
    :param list_: input array.
    :param datatype: type of data in array (should be np.int or np.float).
    :return: list of Box type.
    """
    if len(list_) != 4:
        raise TypeError("List length must be equal 4")
    return Box(datatype(list_))


if __name__ == "__main__":
    res = BoundingBox((20, 40), list_to_box([2, 3, 12, 25]), "coco")
    print(res)
