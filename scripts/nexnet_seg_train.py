import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import glob
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from models import AutoencoderMaSA, NexNet_Seg, NexNet_Seg_ssl
from utils import add_noise, display_images, img_augmentation, preprocess_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train NexNet models for segmentation.")
parser.add_argument('--model_type', type=str, choices=['ssl', 'no_ssl'], default='ssl',
                    help="Choose model type: 'ssl' for NexNet_Seg_ssl, 'no_ssl' for NexNet_Seg")
parser.add_argument('--dataset', type=str, choices=['CVC', 'Kvasir', 'PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018'], default='CVC',
                    help="Choose dataset: 'CVC', 'Kvasir', 'PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018'")
parser.add_argument('--loss_type', type=str, choices=['dice_bce'], default='dice_bce',
                    help="Choose loss function: 'dice_bce' for Dice-BCE")
parser.add_argument('--freeze_encoder', action='store_true', default=False,
                    help="Freeze the encoder weights during training")
parser.add_argument('--pos_weight', type=float, default=None,
                    help="Positive class weight for BCE loss (computed dynamically if None)")
parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate for decoder layers")
parser.add_argument('--gamma', type=float, default=0.8, help="Gamma parameter for MaSA spatial decay")
parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
parser.add_argument('--num_epochs', type=int, default=250, help="Number of epochs for training (default: 250)")
parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for training (default: 0.0001)")
args = parser.parse_args()

# Configuration
size = 224
batch_size = args.batch_size
num_epochs = args.num_epochs
dataset_name = args.dataset
learning_rate = args.learning_rate
output_dir = os.path.join("outputs", f"{dataset_name}_{args.model_type}")
plot_dir = "plots"
preprocessed_dir = "preprocessed"

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define dataset paths
dataset_paths = {
    'CVC': {
        'root': "../datasets/CVC_CLINICDB/PNG",
        'images': "Original",
        'masks': "Ground Truth",
        'extension': "*.png",
        'has_subdirs': False
    },
    'Kvasir': {
        'root': "../datasets/kvasir_seg",
        'images_train': "Train/images",
        'masks_train': "Train/masks",
        'images_test': "Test/images",
        'masks_test': "Test/masks",
        'extension': "*.jpg",
        'has_subdirs': True
    },
    'PH2': {
        'root': "../datasets/PH2/PH2Dataset/PH2 Dataset images",
        'images_pattern': "**/*Dermoscopic_Image/*.bmp",
        'masks_pattern': "**/*lesion/*.bmp",
        'extension': "*.bmp",
        'has_subdirs': True
    },
    'ISIC2016': {
        'root': "../datasets/ISIC2016",
        'images_train': "ISBI2016_ISIC_Part1_Training_Data",
        'images_test': "ISBI2016_ISIC_Part1_Test_Data",
        'masks_train': "ISBI2016_ISIC_Part1_Training_GroundTruth",
        'masks_test': "ISBI2016_ISIC_Part1_Test_GroundTruth",
        'extension': "*.jpg",
        'has_subdirs': True
    },
    'ISIC2017': {
        'root': "../datasets/ISIC2017",
        'images_train': "ISIC-2017_Training_Data",
        'images_test': "ISIC-2017_Test_Data",
        'images_val': "ISIC-2017_Validation_Data",
        'masks_train': "ISIC-2017_Training_GroundTruth",
        'masks_test': "ISIC-2017_Test_GroundTruth",
        'masks_val': "ISIC-2017_Validation_GroundTruth",
        'extension': "*.jpg",
        'has_subdirs': True
    },
    'ISIC2018': {
        'root': "../datasets/ISIC2018",
        'images_train': "ISIC2018_Task1-2_Training_Input",
        'images_test': "ISIC2018_Task1-2_Test_Input",
        'images_val': "ISIC2018_Task1-2_Validation_Input",
        'masks_train': "ISIC2018_Task1_Training_GroundTruth",
        'masks_test': "ISIC2018_Task1_Test_GroundTruth",
        'masks_val': "ISIC2018_Task1_Validation_GroundTruth",
        'extension': "*.jpg",
        'has_subdirs': True
    }
}

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = np.array(Image.open(self.image_paths[idx]).resize((size, size)), dtype=np.float32)
            mask = np.array(Image.open(self.mask_paths[idx]).resize((size, size)), dtype=np.float32)

            logger.debug(f"Image {idx} original range: min={np.min(image)}, max={np.max(image)}")
            logger.debug(f"Mask {idx} original range: min={np.min(mask)}, max={np.max(mask)}")

            if np.max(image) > 1.0:
                image = image / 255.0
            image = min_max_scaler_image(image)

            if np.max(mask) > 1.0:
                mask = mask / 255.0
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0.5).astype(np.float32)

            if np.isnan(image).any() or np.isinf(image).any():
                logger.warning(f"NaN or infinite values in image at index {idx}: {self.image_paths[idx]}")
                raise ValueError("Invalid image data")
            if np.isnan(mask).any() or np.isinf(mask).any():
                logger.warning(f"NaN or infinite values in mask at index {idx}: {self.mask_paths[idx]}")
                raise ValueError("Invalid mask data")

            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)

            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            if self.transform:
                image = self.transform(image)
                torch.manual_seed(seed)
                mask = self.transform(mask)

            return image, mask
        except Exception as e:
            logger.error(f"Error loading data at index {idx}: {e}")
            raise e

def min_max_scaler_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)

# Compute class imbalance for pos_weight
def compute_class_imbalance(mask_data):
    # Ensure mask_data is a tensor
    if not isinstance(mask_data, torch.Tensor):
        mask_data = torch.tensor(mask_data, dtype=torch.float32)
    
    # Squeeze the channel dimension if present
    mask = mask_data.squeeze(1) if mask_data.dim() == 4 else mask_data  # (N, 1, H, W) -> (N, H, W)
    
    # Threshold to get binary values
    foreground = (mask > 0.5).float()  # 1.0 where mask > 0.5, 0.0 otherwise
    background = (mask <= 0.5).float()  # 1.0 where mask <= 0.5, 0.0 otherwise
    
    # Sum over all pixels across all samples
    total_foreground = torch.sum(foreground).item()
    total_background = torch.sum(background).item()
    
    if total_foreground == 0:
        logger.warning("No foreground pixels found in the dataset. Setting pos_weight to 1.0.")
        return 1.0
    
    pos_weight = total_background / total_foreground
    logger.info(f"Computed class imbalance: foreground pixels = {total_foreground}, "
                f"background pixels = {total_background}, pos_weight = {pos_weight:.4f}")
    return pos_weight

# Dataset validation and sanity check
def validate_dataset_readiness(dataset, sample_size=5):
    logger.info(f"Validating dataset readiness with {sample_size} samples...")
    for i in range(min(sample_size, len(dataset))):
        try:
            image, mask = dataset[i]
            if not (torch.all(image >= 0) and torch.all(image <= 1)):
                logger.error(f"Image {i} values out of range [0, 1]: min={torch.min(image).item()}, max={torch.max(image).item()}")
                raise ValueError(f"Image {i} values out of range [0, 1]")
            if not (torch.all(mask >= 0) and torch.all(mask <= 1)):
                logger.error(f"Mask {i} values out of range [0, 1]: min={torch.min(mask).item()}, max={torch.max(mask).item()}")
                raise ValueError(f"Mask {i} values out of range [0, 1]")
        except Exception as e:
            logger.error(f"Validation failed at index {i}: {e}")
            raise e
    logger.info("Dataset validation completed successfully. All values are in [0, 1].")

def save_sanity_check_plots(dataset, save_dir="sanity_checks", num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        image = image.permute(1, 2, 0).cpu().numpy()
        mask = mask.squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Image {i}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f"sanity_check_{i}.png"))
        plt.close()

# Load dataset based on dataset type colon
def load_dataset(dataset_name):
    config = dataset_paths[dataset_name]
    root = config['root']
    extension = config['extension']
    
    if dataset_name in ['CVC', 'Kvasir']:
        if config['has_subdirs']:
            images_train = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['images_train'], extension))))
            masks_train = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['masks_train'], extension))))
            images_test = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['images_test'], extension))))
            masks_test = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['masks_test'], extension))))
            
            imgs_path_list = images_train + images_test
            masks_path_list = masks_train + masks_test
        else:
            imgs_path_list = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['images'], extension))))
            masks_path_list = sorted(filter(os.path.isfile, glob.glob(os.path.join(root, config['masks'], extension))))
    logger.info(f"Found {len(imgs_path_list)} images and {len(masks_path_list)} masks for {dataset_name}")
    return imgs_path_list, masks_path_list

# Data loading function with parallel processing
def load_image_batch(args):
    img_path, mask_path, size = args
    try:
        img = np.array(Image.open(img_path).resize((size, size)), dtype=np.float32) / 255.0
        if mask_path:
            mask = np.array(Image.open(mask_path).resize((size, size)), dtype=np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = mask[:, :, 0] if mask.ndim == 3 else mask  # Take first channel if RGB
            return img, mask
        return img, None
    except Exception as e:
        logger.error(f"Error loading {img_path}: {e}")
        return None, None

def load_images_parallel(image_paths, mask_paths=None, num_processes=mp.cpu_count()):
    if mask_paths:
        if not image_paths or not mask_paths:
            logger.error(f"No image or mask paths found: images={len(image_paths)}, masks={len(mask_paths)}")
            return np.array([]), np.array([])
        args = [(img_path, mask_path, size) for img_path, mask_path in zip(image_paths, mask_paths)]
    else:
        if not image_paths:
            logger.error("No image paths found")
            return np.array([]), np.array([])
        args = [(img_path, None, size) for img_path in image_paths]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(load_image_batch, args)
    
    # Filter out None results and unpack
    valid_results = [(r[0], r[1]) for r in results if r[0] is not None and (r[1] is not None if mask_paths else True)]
    if not valid_results:
        logger.error("No valid image-mask pairs loaded")
        return np.array([]), np.array([]) if mask_paths else np.array([])
    
    images, masks = zip(*valid_results)
    return np.array(images), np.array(masks) if mask_paths else np.array(images)

# Prepare skin datasets with caching
def prepare_skin_data():
    preprocessed_dir = "preprocessed"
    cache_files = {
        'PH2': {'X_train': "X_train_PH.npy", 'y_train': "y_train_PH.npy", 'X_val': "X_val_PH.npy", 'y_val': "y_val_PH.npy", 'X_test': "X_test_PH.npy", 'y_test': "y_test_PH.npy"},
        'ISIC2016': {'X_train': "X_train_I16.npy", 'y_train': "y_train_I16.npy", 'X_val': "X_val_I16.npy", 'y_val': "y_val_I16.npy", 'X_test': "X_test_I16.npy", 'y_test': "y_test_I16.npy"},
        'ISIC2017': {'X_train': "X_train_I17.npy", 'y_train': "y_train_I17.npy", 'X_val': "X_val_I17.npy", 'y_val': "y_val_I17.npy", 'X_test': "X_test_I17.npy", 'y_test': "y_test_I17.npy"},
        'ISIC2018': {'X_train': "X_train_I18.npy", 'y_train': "y_train_I18.npy", 'X_val': "X_val_I18.npy", 'y_val': "y_val_I18.npy", 'X_test': "X_test_I18.npy", 'y_test': "y_test_I18.npy"}
    }

    # Load preprocessed data if available
    all_loaded = True
    for file in cache_files[dataset_name].values():
        cache_file = os.path.join(preprocessed_dir, file)
        if not os.path.exists(cache_file):
            all_loaded = False
            break

    if all_loaded:
        logger.info("Loading preprocessed data from cache")
        X_train = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_train']))
        y_train = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_train']))
        X_val = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_val']))
        y_val = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_val']))
        X_test = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_test']))
        y_test = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_test']))
    else:
        logger.info("Preparing data from scratch")
        config = dataset_paths[dataset_name]
        root = config['root']
        extension = config['extension']

        if dataset_name == 'PH2':
            imgs_path_list = sorted(glob.glob(os.path.join(root, config['images_pattern']), recursive=True))
            masks_path_list = sorted(glob.glob(os.path.join(root, config['masks_pattern']), recursive=True))
            if not imgs_path_list or not masks_path_list:
                logger.error(f"PH2 dataset not found. Checked paths: images={imgs_path_list}, masks={masks_path_list}")
                logger.info("Attempting broader search for PH2 files...")
                imgs_path_list = sorted(glob.glob(os.path.join(root, "**", "*.bmp"), recursive=True))
                masks_path_list = sorted(glob.glob(os.path.join(root, "**", "*_lesion.bmp"), recursive=True))
                if not imgs_path_list or not masks_path_list:
                    raise FileNotFoundError("PH2 dataset files not found after broader search. Please check the directory structure.")
        elif dataset_name == 'ISIC2016':
            imgs_path_list_tr = sorted(glob.glob(os.path.join(root, config['images_train'], "*.jpg")))
            imgs_path_list_ts = sorted(glob.glob(os.path.join(root, config['images_test'], "*.jpg")))
            masks_path_list_tr = sorted(glob.glob(os.path.join(root, config['masks_train'], "*.png")))
            masks_path_list_ts = sorted(glob.glob(os.path.join(root, config['masks_test'], "*.png")))
            imgs_path_list = imgs_path_list_tr + imgs_path_list_ts
            masks_path_list = masks_path_list_tr + masks_path_list_ts
        elif dataset_name == 'ISIC2017':
            imgs_path_list_tr = sorted(glob.glob(os.path.join(root, config['images_train'], "*.jpg")))
            imgs_path_list_ts = sorted(glob.glob(os.path.join(root, config['images_test'], "*.jpg")))
            imgs_path_list_val = sorted(glob.glob(os.path.join(root, config['images_val'], "*.jpg")))
            masks_path_list_tr = sorted(glob.glob(os.path.join(root, config['masks_train'], "*.png")))
            masks_path_list_ts = sorted(glob.glob(os.path.join(root, config['masks_test'], "*.png")))
            masks_path_list_val = sorted(glob.glob(os.path.join(root, config['masks_val'], "*.png")))
            imgs_path_list = imgs_path_list_tr + imgs_path_list_ts + imgs_path_list_val
            masks_path_list = masks_path_list_tr + masks_path_list_ts + masks_path_list_val
        elif dataset_name == 'ISIC2018':
            imgs_path_list_tr = sorted(glob.glob(os.path.join(root, config['images_train'], "*.jpg")))
            imgs_path_list_ts = sorted(glob.glob(os.path.join(root, config['images_test'], "*.jpg")))
            imgs_path_list_val = sorted(glob.glob(os.path.join(root, config['images_val'], "*.jpg")))
            masks_path_list_tr = sorted(glob.glob(os.path.join(root, config['masks_train'], "*.png")))
            masks_path_list_ts = sorted(glob.glob(os.path.join(root, config['masks_test'], "*.png")))
            masks_path_list_val = sorted(glob.glob(os.path.join(root, config['masks_val'], "*.png")))
            imgs_path_list = imgs_path_list_tr + imgs_path_list_ts + imgs_path_list_val
            masks_path_list = masks_path_list_tr + masks_path_list_ts + masks_path_list_val

        imgs_arr, masks_arr = load_images_parallel(imgs_path_list, masks_path_list)
        if imgs_arr.size == 0:
            raise ValueError(f"No valid images loaded for {dataset_name} dataset")

        X_train, X_test, y_train, y_test = train_test_split(imgs_arr, masks_arr, test_size=0.25, random_state=101)
        logger.info(f"{dataset_name} Train (before split): {X_train.shape}, Test: {X_test.shape}")

        x_rotated, y_rotated, x_flipped, y_flipped = img_augmentation(X_train, y_train)
        X_train_full = np.concatenate([X_train, x_rotated, x_flipped])
        y_train_full = np.concatenate([y_train, y_rotated, y_flipped])
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=101)

        # Save preprocessed data as NumPy arrays
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_train']), X_train)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_train']), y_train)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_val']), X_val)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_val']), y_val)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_test']), X_test)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_test']), y_test)

    # Process clean images and masks (convert to tensors here)
    X_train_processed = preprocess_images(X_train)  # Returns tensor in (N, C, H, W) for images
    X_val_processed = preprocess_images(X_val)
    X_test_processed = preprocess_images(X_test)
    y_train_processed = preprocess_images(y_train)  # Returns tensor, but likely in wrong shape for masks
    y_val_processed = preprocess_images(y_val)
    y_test_processed = preprocess_images(y_test)

    # Correct mask shapes: expected (N, 1, H, W)
    if y_train_processed.dim() == 4:
        if y_train_processed.size(0) == 1 and y_train_processed.size(1) == size:
            y_train_processed = y_train_processed.permute(2, 0, 1, 3)  # (N, 1, 224, 224)
        elif y_train_processed.size(0) == 1 and y_train_processed.size(2) == size:
            y_train_processed = y_train_processed.permute(1, 0, 2, 3)

    if y_val_processed.dim() == 4:
        if y_val_processed.size(0) == 1 and y_val_processed.size(1) == size:
            y_val_processed = y_val_processed.permute(2, 0, 1, 3)
        elif y_val_processed.size(0) == 1 and y_val_processed.size(2) == size:
            y_val_processed = y_val_processed.permute(1, 0, 2, 3)

    if y_test_processed.dim() == 4:
        if y_test_processed.size(0) == 1 and y_test_processed.size(1) == size:
            y_test_processed = y_test_processed.permute(2, 0, 1, 3)
        elif y_test_processed.size(0) == 1 and y_test_processed.size(2) == size:
            y_test_processed = y_test_processed.permute(1, 0, 2, 3)

    # Ensure images are in (N, C, H, W) format
    if X_train_processed.dim() == 4 and X_train_processed.size(1) != 3:
        X_train_processed = X_train_processed.permute(0, 3, 1, 2)
    if X_val_processed.dim() == 4 and X_val_processed.size(1) != 3:
        X_val_processed = X_val_processed.permute(0, 3, 1, 2)
    if X_test_processed.dim() == 4 and X_test_processed.size(1) != 3:
        X_test_processed = X_test_processed.permute(0, 3, 1, 2)

    logger.info(f"X_train_{dataset_name}_processed shape: {X_train_processed.shape}, min: {X_train_processed.min().item()}, max: {X_train_processed.max().item()}")
    logger.info(f"y_train_{dataset_name}_processed shape: {y_train_processed.shape}, min: {y_train_processed.min().item()}, max: {y_train_processed.max().item()}")
    logger.info(f"X_val_{dataset_name}_processed shape: {X_val_processed.shape}, min: {X_val_processed.min().item()}, max: {X_val_processed.max().item()}")
    logger.info(f"y_val_{dataset_name}_processed shape: {y_val_processed.shape}, min: {y_val_processed.min().item()}, max: {y_val_processed.max().item()}")
    logger.info(f"X_test_{dataset_name}_processed shape: {X_test_processed.shape}, min: {X_test_processed.min().item()}, max: {X_test_processed.max().item()}")
    logger.info(f"y_test_{dataset_name}_processed shape: {y_test_processed.shape}, min: {y_test_processed.min().item()}, max: {y_test_processed.max().item()}")

    logger.info("Data preparation complete")
    return (X_train_processed, y_train_processed), (X_val_processed, y_val_processed), (X_test_processed, y_test_processed)

# Prepare colon datasets with caching
def prepare_colon_data():
    preprocessed_dir = "preprocessed"
    cache_files = {
        'CVC': {'X_train': "X_train_CVC.npy", 'y_train': "y_train_CVC.npy", 'X_val': "X_val_CVC.npy", 'y_val': "y_val_CVC.npy", 'X_test': "X_test_CVC.npy", 'y_test': "y_test_CVC.npy"},
        'Kvasir': {'X_train': "X_train_Kvasir.npy", 'y_train': "y_train_Kvasir.npy", 'X_val': "X_val_Kvasir.npy", 'y_val': "y_val_Kvasir.npy", 'X_test': "X_test_Kvasir.npy", 'y_test': "y_test_Kvasir.npy"}
    }

    # Load preprocessed data if available
    all_loaded = True
    for file in cache_files[dataset_name].values():
        cache_file = os.path.join(preprocessed_dir, file)
        if not os.path.exists(cache_file):
            all_loaded = False
            break

    if all_loaded:
        logger.info("Loading preprocessed data from cache")
        X_train = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_train']))
        y_train = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_train']))
        X_val = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_val']))
        y_val = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_val']))
        X_test = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_test']))
        y_test = np.load(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_test']))
    else:
        logger.info("Preparing data from scratch")
        imgs_path_list, masks_path_list = load_dataset(dataset_name)
        if not imgs_path_list or not masks_path_list:
            raise FileNotFoundError(f"No images or masks found for {dataset_name} dataset")

        imgs_arr, masks_arr = load_images_parallel(imgs_path_list, masks_path_list)
        if imgs_arr.size == 0:
            raise ValueError(f"No valid images loaded for {dataset_name} dataset")

        X_train, X_test, y_train, y_test = train_test_split(imgs_arr, masks_arr, test_size=0.25, random_state=101)
        logger.info(f"{dataset_name} Train (before split): {X_train.shape}, Test: {X_test.shape}")

        x_rotated, y_rotated, x_flipped, y_flipped = img_augmentation(X_train, y_train)
        X_train_full = np.concatenate([X_train, x_rotated, x_flipped])
        y_train_full = np.concatenate([y_train, y_rotated, y_flipped])
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=101)

        # Save preprocessed data as NumPy arrays
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_train']), X_train)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_train']), y_train)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_val']), X_val)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_val']), y_val)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['X_test']), X_test)
        np.save(os.path.join(preprocessed_dir, cache_files[dataset_name]['y_test']), y_test)

    # Process clean images and masks (convert to tensors here)
    X_train_processed = preprocess_images(X_train)  # Returns tensor in (N, C, H, W) for images
    X_val_processed = preprocess_images(X_val)
    X_test_processed = preprocess_images(X_test)
    y_train_processed = preprocess_images(y_train)  # Returns tensor, but adjust shape for masks
    y_val_processed = preprocess_images(y_val)
    y_test_processed = preprocess_images(y_test)

    # Correct mask shapes: expected (N, 1, H, W)
    if y_train_processed.dim() == 4:
        if y_train_processed.size(0) == 1 and y_train_processed.size(1) == size:
            y_train_processed = y_train_processed.permute(2, 0, 1, 3)  # (N, 1, 224, 224)
        elif y_train_processed.size(0) == 1 and y_train_processed.size(2) == size:
            y_train_processed = y_train_processed.permute(1, 0, 2, 3)

    if y_val_processed.dim() == 4:
        if y_val_processed.size(0) == 1 and y_val_processed.size(1) == size:
            y_val_processed = y_val_processed.permute(2, 0, 1, 3)
        elif y_val_processed.size(0) == 1 and y_val_processed.size(2) == size:
            y_val_processed = y_val_processed.permute(1, 0, 2, 3)

    if y_test_processed.dim() == 4:
        if y_test_processed.size(0) == 1 and y_test_processed.size(1) == size:
            y_test_processed = y_test_processed.permute(2, 0, 1, 3)
        elif y_test_processed.size(0) == 1 and y_test_processed.size(2) == size:
            y_test_processed = y_test_processed.permute(1, 0, 2, 3)

    # Ensure images are in (N, C, H, W) format
    if X_train_processed.dim() == 4 and X_train_processed.size(1) != 3:
        X_train_processed = X_train_processed.permute(0, 3, 1, 2)
    if X_val_processed.dim() == 4 and X_val_processed.size(1) != 3:
        X_val_processed = X_val_processed.permute(0, 3, 1, 2)
    if X_test_processed.dim() == 4 and X_test_processed.size(1) != 3:
        X_test_processed = X_test_processed.permute(0, 3, 1, 2)

    logger.info(f"X_train_{dataset_name}_processed shape: {X_train_processed.shape}, min: {X_train_processed.min().item()}, max: {X_train_processed.max().item()}")
    logger.info(f"y_train_{dataset_name}_processed shape: {y_train_processed.shape}, min: {y_train_processed.min().item()}, max: {y_train_processed.max().item()}")
    logger.info(f"X_val_{dataset_name}_processed shape: {X_val_processed.shape}, min: {X_val_processed.min().item()}, max: {X_val_processed.max().item()}")
    logger.info(f"y_val_{dataset_name}_processed shape: {y_val_processed.shape}, min: {y_val_processed.min().item()}, max: {y_val_processed.max().item()}")
    logger.info(f"X_test_{dataset_name}_processed shape: {X_test_processed.shape}, min: {X_test_processed.min().item()}, max: {X_test_processed.max().item()}")
    logger.info(f"y_test_{dataset_name}_processed shape: {y_test_processed.shape}, min: {y_test_processed.min().item()}, max: {y_test_processed.max().item()}")

    logger.info("Data preparation complete")
    return (X_train_processed, y_train_processed), (X_val_processed, y_val_processed), (X_test_processed, y_test_processed)

# Load dataset
if dataset_name in ['PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018']:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_skin_data()
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Size mismatch: X_train has {X_train.shape[0]} samples, y_train has {y_train.shape[0]} samples")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
elif dataset_name in ['CVC', 'Kvasir']:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_colon_data()
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Size mismatch: X_train has {X_train.shape[0]} samples, y_train has {y_train.shape[0]} samples")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

# Compute pos_weight based on training masks
if args.pos_weight is None:
    pos_weight = compute_class_imbalance(y_train)
else:
    pos_weight = args.pos_weight
    logger.info(f"Using user-specified pos_weight: {pos_weight:.4f}")

# Create DataLoaders
if dataset_name in ['PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018', 'CVC', 'Kvasir']:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Model setup
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = size, size, 3
num_labels = 1

# Select autoencoder model based on dataset
if dataset_name in ['PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018']:
    model_path = f"models/skin_autoencoder.pt"
else:
    model_path = f"models/colon_autoencoder.pt"

auto_enc = AutoencoderMaSA(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, "autoencoder_masa", use_masa=True, gamma=args.gamma).to(device)
if os.path.exists(model_path):
    auto_enc.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info(f"Loaded pre-trained autoencoder weights from {model_path}")
else:
    logger.warning(f"Autoencoder weights not found at {model_path}. Training from scratch.")

if args.model_type == 'ssl':
    model = NexNet_Seg_ssl(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        filters=[32, 64, 128, 256, 256, 512],
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        depth=1,
        num_classes=num_labels,
        dataset=dataset_name,
        encoder=auto_enc,
        dr=args.dropout_rate,
        use_masa=True,
        gamma=args.gamma,
        freeze_encoder=args.freeze_encoder
    ).to(device)
else:
    model = NexNet_Seg(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        filters=[32, 64, 128, 256, 256, 512],
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        depth=1,
        num_classes=num_labels,
        dataset=dataset_name,
        dr=args.dropout_rate,
        use_masa=True,
        gamma=args.gamma
    ).to(device)

# Loss function and metrics
def dice_bce_loss(outputs, targets, pos_weight=5.0, smooth=1.0):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(outputs.device))
    bce_loss = bce(outputs, targets)
    
    outputs = torch.sigmoid(outputs)
    outputs_flat = outputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (outputs_flat * targets_flat).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (outputs_flat.sum() + targets_flat.sum() + smooth)
    
    return bce_loss + dice_loss

def compute_metrics(outputs, targets, thresholds=np.arange(0.3, 0.7, 0.05), smooth=1.0):
    best_threshold = 0.5
    best_dice = 0.0
    with torch.no_grad():
        for thresh in thresholds:
            preds = (torch.sigmoid(outputs) > thresh).float()
            targets_flat = targets.view(-1)
            preds_flat = preds.view(-1)
            TP = (preds_flat * targets_flat).sum()
            dice = (2. * TP + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
            if dice > best_dice:
                best_dice = dice
                best_threshold = thresh
    preds = (torch.sigmoid(outputs) > best_threshold).float()
    targets = targets.float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    TP = (preds_flat * targets_flat).sum()
    FP = (preds_flat * (1 - targets_flat)).sum()
    FN = ((1 - preds_flat) * targets_flat).sum()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    return precision.item(), recall.item(), iou.item(), dice.item(), best_threshold

# Training setup
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

# Track best metrics
best_train_loss = float('inf')
best_train_dice = 0.0
best_val_loss = float('inf')
best_val_dice = 0.0
best_model_path = f"models/nexnet_seg_{args.model_type}_best_{dataset_name}.pt"

# Lists to store metrics for plotting
train_losses, val_losses = [], []
train_dices, val_dices = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_ious, val_ious = [], []

start_time = datetime.now()
warmup_epochs = args.warmup_epochs
patience = 30
epochs_no_improve = 0

for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        lr = learning_rate * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    model.train()
    train_loss, train_dice, train_precision, train_recall, train_iou = 0.0, 0.0, 0.0, 0.0, 0.0
    train_batches = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_bce_loss(outputs, masks, pos_weight=pos_weight)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping batch {batch_idx} due to NaN or inf loss")
            continue
        
        precision, recall, iou, dice, _ = compute_metrics(outputs, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        train_loss += loss.item()
        train_dice += dice
        train_precision += precision
        train_recall += recall
        train_iou += iou
        train_batches += 1
        
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} [Training]: {batch_idx+1}/{len(train_loader)}, "
                   f"Loss={loss.item():.4f}, Dice={dice:.4f}, Precision={precision:.4f}, "
                   f"Recall={recall:.4f}, IoU={iou:.4f}")
    
    if train_batches == 0:
        logger.warning(f"Epoch {epoch+1} skipped all batches due to NaN/inf losses")
        continue
    
    train_loss /= train_batches
    train_dice /= train_batches
    train_precision /= train_batches
    train_recall /= train_batches
    train_iou /= train_batches
    
    # Update best train metrics
    if train_loss < best_train_loss:
        best_train_loss = train_loss
    if train_dice > best_train_dice:
        best_train_dice = train_dice
    
    # Validation
    model.eval()
    val_loss, val_dice, val_precision, val_recall, val_iou = 0.0, 0.0, 0.0, 0.0, 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = dice_bce_loss(outputs, masks, pos_weight=pos_weight)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping validation batch {batch_idx} due to NaN or inf loss")
                continue
            
            precision, recall, iou, dice, _ = compute_metrics(outputs, masks)
            
            val_loss += loss.item()
            val_dice += dice
            val_precision += precision
            val_recall += recall
            val_iou += iou
            val_batches += 1
            
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} [Validation]: {batch_idx+1}/{len(val_loader)}, "
                       f"Loss={loss.item():.4f}, Dice={dice:.4f}, Precision={precision:.4f}, "
                       f"Recall={recall:.4f}, IoU={iou:.4f}")
    
    if val_batches == 0:
        logger.warning(f"Epoch {epoch+1} skipped all validation batches due to NaN/inf losses")
        continue
    
    val_loss /= val_batches
    val_dice /= val_batches
    val_precision /= val_batches
    val_recall /= val_batches
    val_iou /= val_batches
    
    # Update best val metrics and save model
    if val_dice > best_val_dice and not np.isnan(val_dice):
        best_val_dice = val_dice
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1}/{num_epochs}: Saved best model with val_dice: {val_dice:.4f}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Saved best model with val_dice: {val_dice:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        logger.debug(f"Epoch {epoch+1}/{num_epochs}: val_dice {val_dice:.4f} not better than best_val_dice {best_val_dice:.4f}")
    
    # Store metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)
    train_precisions.append(train_precision)
    val_precisions.append(val_precision)
    train_recalls.append(train_recall)
    val_recalls.append(val_recall)
    train_ious.append(train_iou)
    val_ious.append(val_iou)
    
    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}, "
                f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, "
                f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, "
                f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
    scheduler.step(val_dice)

# Testing loop and save predictions
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
test_loss, test_dice, test_precision, test_recall, test_iou = 0.0, 0.0, 0.0, 0.0, 0.0
test_batches = 0

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = dice_bce_loss(outputs, masks, pos_weight=pos_weight)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping test batch {batch_idx} due to NaN or inf loss")
            continue
        
        precision, recall, iou, dice, _ = compute_metrics(outputs, masks)
        
        test_loss += loss.item()
        test_dice += dice
        test_precision += precision
        test_recall += recall
        test_iou += iou
        test_batches += 1
        
        # Save input image, mask, and prediction
        for i in range(images.size(0)):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            msk = masks[i].squeeze(0).cpu().numpy()
            pred = torch.sigmoid(outputs[i]).squeeze(0).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            
            img_name = f"img_{i}_input.png"
            msk_name = f"msk_{i}_mask.png"
            pred_name = f"pred_{i}_pred.png"
            
            plt.imsave(os.path.join(output_dir, img_name), img)
            plt.imsave(os.path.join(output_dir, msk_name), msk, cmap='gray')
            plt.imsave(os.path.join(output_dir, pred_name), pred_binary, cmap='gray')
        
        tqdm.write(f"Testing: {batch_idx+1}/{len(test_loader)}, "
                   f"Loss={loss.item():.4f}, Dice={dice:.4f}, Precision={precision:.4f}, "
                   f"Recall={recall:.4f}, IoU={iou:.4f}")

if test_batches == 0:
    logger.warning("No valid test batches processed due to NaN/inf losses")
else:
    test_loss /= test_batches
    test_dice /= test_batches
    test_precision /= test_batches
    test_recall /= test_batches
    test_iou /= test_batches

    logger.info(f"Best Final Training Performance: "
                f"Train Loss: {best_train_loss:.4f}, Train Dice: {best_train_dice:.4f}")
    logger.info(f"Best Final Validation Performance: "
                f"Val Loss: {best_val_loss:.4f}, Val Dice: {best_val_dice:.4f}")
    logger.info(f"Final Testing Performance with Best Model: "
                f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}, "
                f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test IoU: {test_iou:.4f}")

# Plot training and validation progress
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_dices, label='Train Dice')
plt.plot(val_dices, label='Val Dice')
plt.title('Dice Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_precisions, label='Train Precision')
plt.plot(val_precisions, label='Val Precision')
plt.title('Precision Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(train_ious, label='Train IoU')
plt.plot(val_ious, label='Val IoU')
plt.title('IoU Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"training_progress_{dataset_name}_{args.model_type}.png"))
plt.close()

execution_time = datetime.now() - start_time
logger.info(f"NexNet_Seg execution time is: {execution_time}")
logger.info("Training and evaluation completed.")
