import numpy as np
from data_generation import generate_examples, CircleParams
from model import create_cnn_model

def create_dataset(num_samples, img_size=100, noise_level=0.5):
    data_gen = generate_examples(noise_level=noise_level, img_size=img_size)
    images, labels = [], []
    for _ in range(num_samples):
        img, params = next(data_gen)
        images.append(img.reshape(img_size, img_size, 1))  # reshaping img to match CNN
        labels.append([params.row, params.col, params.radius])
    return np.array(images), np.array(labels)

def train_and_evaluate():
    num_train_samples = 5000
    num_val_samples = 1000
    train_images, train_labels = create_dataset(num_train_samples)
    val_images, val_labels = create_dataset(num_val_samples)

    model = create_cnn_model((100, 100, 1))
    model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    # Evaluate the model
    test_images, test_labels = create_dataset(num_val_samples)
    predictions = model.predict(test_images)
    iou_scores = [iou(CircleParams(*true), CircleParams(*pred)) for true, pred in zip(test_labels, predictions)]
    average_iou = np.mean(iou_scores)
    print(f"Avg IOU Score: {average_iou}")


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        # If the distance between the centers is greater than the sum of the radii, then the circles don't intersect
        return 0.0
    if d <= abs(r1 - r2):
        # If the distance between the centers is less than the absolute difference of the radii, then one circle is 
        # inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        return smaller_r ** 2 / larger_r ** 2
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union