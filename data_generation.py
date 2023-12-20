import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
from typing import Tuple, Optional, Generator

class CircleParams:
    def __init__(self, row: int, col: int, radius: int):
        self.row = row
        self.col = col
        self.radius = radius

def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img

def noisy_circle(img_size: int, min_radius: float, max_radius: float, noise_level: float) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """
    img = np.zeros((img_size, img_size))
    radius = np.random.randint(min_radius, max_radius)
    row, col = np.random.randint(img_size, size=2)
    draw_circle(img, row, col, radius)
    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise
    return img, CircleParams(row, col, radius)

def generate_examples(noise_level: float = 0.5, img_size: int = 100, min_radius: Optional[int] = None, max_radius: Optional[int] = None) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    while True:
        img, params = noisy_circle(img_size, min_radius, max_radius, noise_level)
        yield img, params


def show_circle(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Circle')
    plt.show()