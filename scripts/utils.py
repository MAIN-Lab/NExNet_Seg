# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def preprocess_images(images):
    """Convert images to PyTorch tensor and ensure proper shape."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()  # No normalization, already in [0, 1]
    elif isinstance(images, torch.Tensor):
        images = images.float()  # Keep as is
    # Ensure channel-first format (N, H, W, C) -> (N, C, H, W)
    if images.dim() == 4 and images.shape[1] not in [1, 3]:  # (N, H, W, C) -> (N, C, H, W)
        images = images.permute(0, 3, 1, 2)
    elif images.dim() == 3:  # (H, W, C) -> (C, H, W)
        images = images.permute(2, 0, 1).unsqueeze(0)
    return images

def add_noise(images, noise_factor=0.1):
    """Add Gaussian noise to images."""
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0., 1.)

def display_images(array1, array2):
    """Display 10 random pairs of images."""
    array1 = array1.cpu().numpy() if isinstance(array1, torch.Tensor) else array1
    array2 = array2.cpu().numpy() if isinstance(array1, torch.Tensor) else array2
    n = 10
    indices = np.random.randint(0, len(array1), size=n)
    images1 = array1[indices]
    images2 = array2[indices]

    plt.figure(figsize=(20, 4))
    for i, (img1, img2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img1.squeeze() if img1.shape[0] == 1 else img1.transpose(1, 2, 0))
        plt.gray() if img1.shape[0] == 1 else None
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(img2.squeeze() if img2.shape[0] == 1 else img2.transpose(1, 2, 0))
        plt.gray() if img2.shape[0] == 1 else None
        ax.axis('off')
    plt.show()

def random_rotation(x_image, y_image):
    """Apply random rotation to image and mask."""
    rows_x, cols_x, chl_x = x_image.shape
    rows_y, cols_y = y_image.shape
    rand_num = np.random.randint(-40, 40)
    M1 = cv2.getRotationMatrix2D((cols_x / 2, rows_x / 2), rand_num, 1)
    M2 = cv2.getRotationMatrix2D((cols_y / 2, rows_y / 2), rand_num, 1)
    x_image = cv2.warpAffine(x_image, M1, (cols_x, rows_x))
    y_image = cv2.warpAffine(y_image.astype(np.float32), M2, (cols_y, rows_y))
    return x_image, y_image.astype(np.int32)

def horizontal_flip(x_image, y_image):
    """Apply horizontal flip to image and mask."""
    x_image = cv2.flip(x_image, 1)
    y_image = cv2.flip(y_image.astype(np.float32), 1)
    return x_image, y_image.astype(np.int32)

def img_augmentation(x_train, y_train):
    """Augment images with rotation and flipping."""
    x_rotat, y_rotat, x_flip, y_flip = [], [], [], []
    for idx in range(len(x_train)):
        x, y = random_rotation(x_train[idx], y_train[idx])
        x_rotat.append(x)
        y_rotat.append(y)
        x, y = horizontal_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
    return (np.array(x_rotat), np.array(y_rotat), np.array(x_flip), np.array(y_flip))

# Metrics (PyTorch versions)
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[1, 2, 3])
    sum_ = torch.sum(y_true ** 2, dim=[1, 2, 3]) + torch.sum(y_pred ** 2, dim=[1, 2, 3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac).mean()

def iou(y_true, y_pred, smooth=100):
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def dice_coe(y_true, y_pred, smooth=100):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    return true_positives / (predicted_positives + 1e-7)

def recall(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    return true_positives / (possible_positives + 1e-7)

def accuracy(y_true, y_pred):
    return torch.mean((y_true == torch.round(y_pred)).float())
