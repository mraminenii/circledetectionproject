from typing import NamedTuple, Optional, Tuple, Generator
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
from train_evaluate import train_and_evaluate
from data_generation import noisy_circle, show_circle

def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()

#three lines of code to produce an image for one to visualize an example noisy image + circle.
example_image, example_params = noisy_circle(
img_size=100, min_radius=10, max_radius=50, noise_level=0.3)
show_circle(example_image)