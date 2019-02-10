import os
import cv2

import numpy as np

from PIL import Image as Img
from typing import List, Tuple, NoReturn
from .BoundingBox import BoundingBox, Box, list_to_box


class Image:

    def __init__(self, bounding_box: np.ndarray or Box = None, box_type: str = "yolo",
                 image_matrix: np.ndarray = None, filename: str = None) -> None:
        bounding_box = [0, 0, 0, 0] if bounding_box is None else bounding_box
        # if filename is not None:
        #     filename = to_jpg(filename)
        self.image_matrix:         np.ndarray = cv2.imread(filename) if image_matrix is None else image_matrix
        self.bounding_box:         BoundingBox = BoundingBox(self.image_matrix.shape[-2::-1], bounding_box, box_type)

    def show_img(self) -> NoReturn:
        """
        Show the image with bounding box.
        :return: None.
        """
        Img.fromarray(self.add_box(self.image_matrix, self.bounding_box.get_box())).show()

    def mirroring(self, axis: int = -1) -> object:
        """
        Mirror the input image and change the bounding box parameters.
        :param axis: 0 - vertical, 1 - horizontal, -1 - double mirroring.
        :return: new image class.
        """
        child_box = np.copy(self.bounding_box.get_box())
        if abs(axis) == 1:
            child_box[::2] = list(map(lambda x: self.bounding_box.get_size()[0] - x - 2,  child_box[-2::-2]))
        if axis <= 0:
            child_box[1::2] = list(map(lambda x: self.bounding_box.get_size()[1] - x - 2, child_box[-1::-2]))
        return Image(child_box, "coco", cv2.flip(self.image_matrix, axis))

    def cropping(self, crop_box: Box) -> object:
        """
        Crop the input image and change the bounding box parameters.
        :param crop_box: cropping coordinates.
        :return: new image class.
        """
        child_matrix = self.image_matrix[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
        child_box = np.copy(self.bounding_box.get_box())
        child_box[::2] = list(map(lambda x: x - crop_box[0], child_box[::2]))
        child_box[1::2] = list(map(lambda x: x - crop_box[1], child_box[1::2]))
        return Image(child_box, "coco", child_matrix)

    def shearing(self, new_size: Tuple[int, int]) -> object:
        """
        Shear the input image and change the bounding box parameters.
        :param new_size: new size parameters.
        :return: new image class.
        """
        child_matrix = cv2.resize(self.image_matrix, new_size)
        child_box = np.copy(self.bounding_box.get_box())
        size = self.bounding_box.get_size()
        child_box[::2] = list(map(lambda x: np.round(x * new_size[0] / size[0]), child_box[::2]))
        child_box[1::2] = list(map(lambda x: np.round(x * new_size[1] / size[1]), child_box[1::2]))
        return Image(child_box, "coco", child_matrix)

    def shifting(self, shifts: Tuple[int, int, int]) -> object:
        """
        Shift color for input image and change the bounding box parameters.
        :param shifts: shifts parameters.
        :return: new image class.
        """
        child_matrix = np.array(self.image_matrix, dtype=np.int16)
        for i in range(len(shifts)):
            child_matrix[:, :, i] = self.to_interval(child_matrix[:, :, i] + shifts[i])
        return Image(self.bounding_box.get_box(), "coco", np.uint8(child_matrix))

    def blurring(self, k_size: Tuple[int, int]) -> object:
        """
        Blur image and change the bounding box parameters.
        :param k_size: shifts parameters.
        :return: new image class.
        """
        child_matrix = cv2.GaussianBlur(self.image_matrix, k_size, 0)
        return Image(self.bounding_box.get_box(), "coco", child_matrix)

    def rotation(self, angle: int) -> object:
        """
        Rotate image and change the bounding box parameters.
        :param angle: shifts parameters.
        :return: new image class.
        """
        sub_angle = np.pi / 180 * angle
        size = self.bounding_box.get_size()
        coefficient = size[0] / (size[0] * np.cos(sub_angle) + size[1] * abs(np.sin(sub_angle)))
        m = cv2.getRotationMatrix2D(tuple(map(lambda x: x / 2, self.bounding_box.get_size())), -angle, coefficient)
        child_matrix = cv2.warpAffine(self.image_matrix, m, self.bounding_box.get_size())
        child_matrix = self.flood_fill(child_matrix,
                                       [(0, 0), (0, size[1] - 1), (size[0] - 1, 0), (size[0] - 1, size[1] - 1)],
                                       new_color=tuple(np.median(self.image_matrix, axis=[0, 1])))
        new_big_box = BoundingBox.get_full_box(self.bounding_box.get_box())
        center_vector = np.array(size) // 2
        m = np.array([[np.cos(sub_angle), - np.sin(sub_angle)], [np.sin(sub_angle), np.cos(sub_angle)]])
        new_bbox = np.round([center_vector + coefficient * m.dot(i - center_vector) for i in new_big_box])
        child_box = list_to_box(BoundingBox.get_max_box(new_bbox))
        return Image(child_box, "coco", child_matrix)

    def save(self, filename: str) -> NoReturn:
        """
        Save image.
        ;:param filename: name of saved image.
        :return: None
        """
        Img.fromarray(self.image_matrix, "RGB").save(filename + ".jpg")

    @staticmethod
    def add_box(image_: np.ndarray, bbox: Box) -> np.ndarray:
        """
        Draw the bounding box on the image.
        :param image_: Image matrix.
        :param bbox: Parameters of bounding box.
        :return: Updated image matrix.
        """
        image = image_.copy()
        for i_ in range(int(bbox[0]), int(bbox[2] + 1)):
            if (0 <= i_ < image.shape[1]) and (0 <= bbox[1] < image.shape[0]):
                image[int(bbox[1]):int(bbox[1]) + 2, i_, :] = [255, 0, 0]
            if (0 <= i_ < image.shape[1]) and (0 <= bbox[3] < image.shape[0]):
                image[int(bbox[3]):int(bbox[3]) + 2, i_, :] = [255, 0, 0]
        for i_ in range(int(bbox[1]), int(bbox[3] + 1)):
            if (0 <= i_ < image.shape[0]) and (0 <= bbox[0] < image.shape[1]):
                image[i_, int(bbox[0]):int(bbox[0]) + 2, :] = [255, 0, 0]
            if (0 <= i_ < image.shape[0]) and (0 <= bbox[2] < image.shape[1]):
                image[i_ + 1, int(bbox[2]):int(bbox[2]) + 2, :] = [255, 0, 0]
        return image

    @staticmethod
    def to_interval(array: np.ndarray):
        """
        Clip the list values in the interval [0, 255].
        :param array: Input array.
        :return: normalized array.
        """
        func = np.vectorize(lambda x: max(0, min(255, x)))
        return func(array)

    @staticmethod
    def flood_fill(image_matrix: np.ndarray, points: List[Tuple[int, int]],
                   new_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Make a flood fill from input point.
        :param image_matrix: image RGB matrix.
        :param points: start point.
        :param new_color: color of filling.
        :return: new RGB matrix.
        """
        high, width = image_matrix.shape[:2]
        mask = np.zeros((high + 2, width + 2), np.uint8)
        for point in points:
            _, image, _, _ = cv2.floodFill(image_matrix, mask, point, new_color, (5,) * 3, (5,) * 3, 8)
            image_matrix = image
        return image_matrix


def to_jpg(filename: str) -> str:
    """
    Convert image to JPEG format
    :param filename: path to image.
    :return: new path to image.
    """
    name, img_type = filename.split(".")
    if img_type != "jpg":
        im = Img.open(filename)
        rgb_im = im.convert('RGB')
        rgb_im.save(name + '.jpg')
        os.remove(filename)
        return name + '.jpg'
    return filename


if __name__ == "__main__":
    img = Image([0, 0, 10, 10], filename="test.jpg", box_type="coco")
    img.show_img()
