import itertools

import numpy as np

from .Image import Image
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, NoReturn, Callable, Any, Tuple, ClassVar


ALL_METHODS = [Image.mirroring, Image.rotation, Image.shearing, Image.shifting, Image.cropping, Image.blurring]


class Augmentation:
    methods:            ClassVar[List[Callable]] = []

    def __init__(self, image: Image, probabilities: List[float] = None, methods_list: List[str] = None,
                 count: int = 1,) -> None:
        self.prob:               List[float] = []
        self.image:              Image = image
        self.count:              int = count
        self.children:           List = []
        self.all_methods:        List = []
        self.methods_count:      int = None

        self.set_methods(methods_list, probabilities)

    def set_methods(self, methods_list: List[str], probabilities: List[float] = None) -> NoReturn:
        if methods_list is None:
            methods_list = ALL_METHODS
        for method in methods_list:
            if method == "mirroring":
                self.methods.append(Image.mirroring)
            elif method == "rotation":
                self.methods.append(Image.rotation)
            elif method == "shearing":
                self.methods.append(Image.shearing)
            elif method == "cropping":
                self.methods.append(Image.cropping)
            elif method == "shifting":
                self.methods.append(Image.shifting)
            elif method == "blurring":
                self.methods.append(Image.blurring)
        self.methods = list(set(self.methods))
        self.all_methods, self.methods_count = self.create_methods(self.methods)
        self.prob = probabilities if probabilities is not None else \
            [1.0 / self.methods_count for _ in range(self.methods_count)]

    @staticmethod
    def get_params(image: Image, method: Callable) -> Any:
        """
        Choice random parameters for each augmentation method.
        :param image: input image.
        :param method: function of method.
        :return: random parameters.
        """
        name = method.__name__
        if name == "shifting":
            return tuple([np.random.randint(-95, 97), np.random.randint(-95, 97), np.random.randint(-95, 97)])
        elif name == "shearing":
            return tuple(np.random.randint(400, 1201, 2))
        elif name == "mirroring":
            return np.random.randint(-1, 2)
        elif name == "cropping":
            box = image.bounding_box.get_box()
            size = image.bounding_box.get_size()
            return [np.random.randint(0, box[0] - 1),
                    np.random.randint(0, box[1] - 1),
                    np.random.randint(box[2] + 1, size[0]),
                    np.random.randint(box[3] + 1, size[1])]
        elif name == "blurring":
            return tuple(2 * np.random.randint(1, 9, 2) + 1)
        elif name == "rotation":
            return np.random.randint(-25, 26) / 2

    def create_children(self) -> NoReturn:
        """
        Create augmentation images and append them to array.
        :return: None.
        """
        with ThreadPoolExecutor(max_workers=6) as executor:
            fit_ = [executor.submit(self.one_method,
                                    self.all_methods[i]) for i in np.random.choice(self.methods_count,
                                                                                   self.count,
                                                                                   p=self.prob)]
            wait(fit_)

    def one_method(self, methods: Tuple[Callable, Callable], count: int = 1) -> NoReturn:
        """
        Apply method and append new image to array of children.
        :param methods: function of method.
        :param count: count of application.
        :return: None.
        """
        for _ in range(count):
            image = self.image
            for method in methods:
                image = method(image, self.get_params(image, method))
            self.children.append(image)

    @staticmethod
    def create_methods(list_: List[Callable]) -> Tuple[List[Tuple[Callable, ...]], int]:
        """
        Create pairs of methods.
        :param list_: array of augmentation methods.
        :return: array of combinations.
        """
        res = list(itertools.combinations(list_, (len(list_) - 1) // 2 + 1))
        return res, len(res)


if __name__ == "__main__":
    img = Image([70, 480, 1552, 620], filename="test.jpg", box_type="coco")
    # img.show_img()
    aug = Augmentation(img, count=4)

    aug.create_children()
    for children in aug.children:
        children.show_img()
