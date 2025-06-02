# train_autoencoder_masa_skin.py
import os
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import AutoencoderMaSA
from utils import preprocess_images, add_noise, display_images, img_augmentation
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from torchvision.models import vgg16

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable for CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
root_PH = "../datasets/PH2/PH2Dataset/PH2 Dataset images"
root_I16 = "../datasets/ISIC2016"
root_I17 = "../datasets/ISIC2017"
root_I18 = "../datasets/ISIC2018"
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("preprocessed", exist_ok=True)  # Directory for preprocessed data

# Hyperparameters
size = 224
dataset_name = "skin"
batch_size = 4
num_epochs = 250
learning_rate = 0.0001
patience = 20
# Allow for potential use of workers (set to 0 for now, can increase later)
num_workers = 0  # Start with 0 to avoid CUDA issues, can adjust to 2 or 4 if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
logger.info(f"PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")

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

# Check if preprocessed data exists and load it, otherwise prepare it
def prepare_data():
    preprocessed_dir = "preprocessed"
    cache_files = {
        "X_train_PH.npy": None, "y_train_PH.npy": None, "X_val_PH.npy": None, "y_val_PH.npy": None,
        "X_test_PH.npy": None, "y_test_PH.npy": None,
        "X_train_I16.npy": None, "y_train_I16.npy": None, "X_val_I16.npy": None, "y_val_I16.npy": None,
        "X_test_I16.npy": None, "y_test_I16.npy": None,
        "X_train_I17.npy": None, "y_train_I17.npy": None, "X_val_I17.npy": None, "y_val_I17.npy": None,
        "X_test_I17.npy": None, "y_test_I17.npy": None,
        "X_train_I18.npy": None, "y_train_I18.npy": None, "X_val_I18.npy": None, "y_val_I18.npy": None,
        "X_test_I18.npy": None, "y_test_I18.npy": None
    }

    # Load preprocessed data if available
    all_loaded = True
    for file in cache_files:
        cache_file = os.path.join(preprocessed_dir, file)
        if os.path.exists(cache_file):
            cache_files[file] = np.load(cache_file)
        else:
            all_loaded = False
            break

    if all_loaded:
        logger.info("Loading preprocessed data from cache")
        X_train_PH, y_train_PH = cache_files["X_train_PH.npy"], cache_files["y_train_PH.npy"]
        X_val_PH, y_val_PH = cache_files["X_val_PH.npy"], cache_files["y_val_PH.npy"]
        X_test_PH, y_test_PH = cache_files["X_test_PH.npy"], cache_files["y_test_PH.npy"]
        X_train_I16, y_train_I16 = cache_files["X_train_I16.npy"], cache_files["y_train_I16.npy"]
        X_val_I16, y_val_I16 = cache_files["X_val_I16.npy"], cache_files["y_val_I16.npy"]
        X_test_I16, y_test_I16 = cache_files["X_test_I16.npy"], cache_files["y_test_I16.npy"]
        X_train_I17, y_train_I17 = cache_files["X_train_I17.npy"], cache_files["y_train_I17.npy"]
        X_val_I17, y_val_I17 = cache_files["X_val_I17.npy"], cache_files["y_val_I17.npy"]
        X_test_I17, y_test_I17 = cache_files["X_test_I17.npy"], cache_files["y_test_I17.npy"]
        X_train_I18, y_train_I18 = cache_files["X_train_I18.npy"], cache_files["y_train_I18.npy"]
        X_val_I18, y_val_I18 = cache_files["X_val_I18.npy"], cache_files["y_val_I18.npy"]
        X_test_I18, y_test_I18 = cache_files["X_test_I18.npy"], cache_files["y_test_I18.npy"]
    else:
        logger.info("Preparing data from scratch")

        # PH2 (Load first as requested)
        imgs_path_list_PH = sorted(glob.glob(os.path.join(root_PH, "**", "*Dermoscopic_Image", "*.bmp"), recursive=True))
        masks_path_list_PH = sorted(glob.glob(os.path.join(root_PH, "**", "*lesion", "*.bmp"), recursive=True))
        
        logger.info(f"PH2 image paths found: {len(imgs_path_list_PH)}")
        logger.info(f"PH2 mask paths found: {len(masks_path_list_PH)}")
        if imgs_path_list_PH:
            logger.info(f"First PH2 image path: {imgs_path_list_PH[0]}")
        if masks_path_list_PH:
            logger.info(f"First PH2 mask path: {masks_path_list_PH[0]}")
        
        if not imgs_path_list_PH or not masks_path_list_PH:
            logger.error(f"PH2 dataset not found. Checked paths: images={imgs_path_list_PH}, masks={masks_path_list_PH}")
            logger.info("Attempting broader search for PH2 files...")
            imgs_path_list_PH = sorted(glob.glob(os.path.join(root_PH, "**", "*.bmp"), recursive=True))
            masks_path_list_PH = sorted(glob.glob(os.path.join(root_PH, "**", "*_lesion.bmp"), recursive=True))
            logger.info(f"Broad search - PH2 image paths found: {len(imgs_path_list_PH)}")
            logger.info(f"Broad search - PH2 mask paths found: {len(masks_path_list_PH)}")
            if imgs_path_list_PH:
                logger.info(f"First broad image path: {imgs_path_list_PH[0]}")
            if masks_path_list_PH:
                logger.info(f"First broad mask path: {masks_path_list_PH[0]}")
            if not imgs_path_list_PH or not masks_path_list_PH:
                raise FileNotFoundError("PH2 dataset files not found after broader search. Please check the directory structure.")

        imgs_arr_PH, masks_arr_PH = load_images_parallel(imgs_path_list_PH, masks_path_list_PH)
        if imgs_arr_PH.size == 0:
            raise ValueError("No valid images loaded for PH2 dataset")
        X_train_PH, X_test_PH, y_train_PH, y_test_PH = train_test_split(
            imgs_arr_PH, masks_arr_PH, test_size=0.25, random_state=101
        )
        logger.info(f"PH2 Train (before split): {X_train_PH.shape}, Test: {X_test_PH.shape}")

        x_rotated_PH, y_rotated_PH, x_flipped_PH, y_flipped_PH = img_augmentation(X_train_PH, y_train_PH)
        X_train_full_PH = np.concatenate([X_train_PH, x_rotated_PH, x_flipped_PH])
        y_train_full_PH = np.concatenate([y_train_PH, y_rotated_PH, y_flipped_PH])
        X_train_PH, X_val_PH, y_train_PH, y_val_PH = train_test_split(
            X_train_full_PH, y_train_full_PH, test_size=0.20, random_state=101
        )

        # ISIC2016
        imgs_path_list_tr_I16 = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Training_Data", "*.jpg")))
        imgs_path_list_ts_I16 = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Test_Data", "*.jpg")))
        masks_path_list_tr_I16 = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Training_GroundTruth", "*.png")))
        masks_path_list_ts_I16 = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Test_GroundTruth", "*.png")))

        X_train_I16, y_train_I16 = load_images_parallel(imgs_path_list_tr_I16, masks_path_list_tr_I16)
        X_test_I16, y_test_I16 = load_images_parallel(imgs_path_list_ts_I16, masks_path_list_ts_I16)
        logger.info(f"ISIC2016 Train: {X_train_I16.shape}, Test: {X_test_I16.shape}")

        x_rotated_I16, y_rotated_I16, x_flipped_I16, y_flipped_I16 = img_augmentation(X_train_I16, y_train_I16)
        X_train_full_I16 = np.concatenate([X_train_I16, x_rotated_I16, x_flipped_I16])
        y_train_full_I16 = np.concatenate([y_train_I16, y_rotated_I16, y_flipped_I16])
        X_train_I16, X_val_I16, y_train_I16, y_val_I16 = train_test_split(
            X_train_full_I16, y_train_full_I16, test_size=0.20, random_state=101
        )

        # ISIC2017
        imgs_path_list_tr_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Training_Data", "*.jpg")))
        imgs_path_list_ts_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Test_Data", "*.jpg")))
        imgs_path_list_val_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Validation_Data", "*.jpg")))
        masks_path_list_tr_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Training_GroundTruth", "*.png")))
        masks_path_list_ts_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Test_GroundTruth", "*.png")))
        masks_path_list_val_I17 = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Validation_GroundTruth", "*.png")))

        X_train_I17, y_train_I17 = load_images_parallel(imgs_path_list_tr_I17, masks_path_list_tr_I17)
        X_test_I17, y_test_I17 = load_images_parallel(imgs_path_list_ts_I17, masks_path_list_ts_I17)
        X_val_I17, y_val_I17 = load_images_parallel(imgs_path_list_val_I17, masks_path_list_val_I17)
        logger.info(f"ISIC2017 Train: {X_train_I17.shape}, Val: {X_val_I17.shape}, Test: {X_test_I17.shape}")

        x_rotated_I17, y_rotated_I17, x_flipped_I17, y_flipped_I17 = img_augmentation(X_train_I17, y_train_I17)
        X_train_I17 = np.concatenate([X_train_I17, x_rotated_I17, x_flipped_I17])
        y_train_I17 = np.concatenate([y_train_I17, y_rotated_I17, y_flipped_I17])

        # ISIC2018
        imgs_path_list_tr_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
        imgs_path_list_ts_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1-2_Test_Input", "*.jpg")))
        imgs_path_list_val_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1-2_Validation_Input", "*.jpg")))
        masks_path_list_tr_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1_Training_GroundTruth", "*.png")))
        masks_path_list_ts_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1_Test_GroundTruth", "*.png")))
        masks_path_list_val_I18 = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1_Validation_GroundTruth", "*.png")))

        X_train_I18, y_train_I18 = load_images_parallel(imgs_path_list_tr_I18, masks_path_list_tr_I18)
        X_test_I18, y_test_I18 = load_images_parallel(imgs_path_list_ts_I18, masks_path_list_ts_I18)
        X_val_I18, y_val_I18 = load_images_parallel(imgs_path_list_val_I18, masks_path_list_val_I18)
        logger.info(f"ISIC2018 Train: {X_train_I18.shape}, Val: {X_val_I18.shape}, Test: {X_test_I18.shape}")

        x_rotated_I18, y_rotated_I18, x_flipped_I18, y_flipped_I18 = img_augmentation(X_train_I18, y_train_I18)
        X_train_I18 = np.concatenate([X_train_I18, x_rotated_I18, x_flipped_I18])
        y_train_I18 = np.concatenate([y_train_I18, y_rotated_I18, y_flipped_I18])

        # Save preprocessed data
        np.save(os.path.join(preprocessed_dir, "X_train_PH.npy"), X_train_PH)
        np.save(os.path.join(preprocessed_dir, "y_train_PH.npy"), y_train_PH)
        np.save(os.path.join(preprocessed_dir, "X_val_PH.npy"), X_val_PH)
        np.save(os.path.join(preprocessed_dir, "y_val_PH.npy"), y_val_PH)
        np.save(os.path.join(preprocessed_dir, "X_test_PH.npy"), X_test_PH)
        np.save(os.path.join(preprocessed_dir, "y_test_PH.npy"), y_test_PH)
        np.save(os.path.join(preprocessed_dir, "X_train_I16.npy"), X_train_I16)
        np.save(os.path.join(preprocessed_dir, "y_train_I16.npy"), y_train_I16)
        np.save(os.path.join(preprocessed_dir, "X_val_I16.npy"), X_val_I16)
        np.save(os.path.join(preprocessed_dir, "y_val_I16.npy"), y_val_I16)
        np.save(os.path.join(preprocessed_dir, "X_test_I16.npy"), X_test_I16)
        np.save(os.path.join(preprocessed_dir, "y_test_I16.npy"), y_test_I16)
        np.save(os.path.join(preprocessed_dir, "X_train_I17.npy"), X_train_I17)
        np.save(os.path.join(preprocessed_dir, "y_train_I17.npy"), y_train_I17)
        np.save(os.path.join(preprocessed_dir, "X_val_I17.npy"), X_val_I17)
        np.save(os.path.join(preprocessed_dir, "y_val_I17.npy"), y_val_I17)
        np.save(os.path.join(preprocessed_dir, "X_test_I17.npy"), X_test_I17)
        np.save(os.path.join(preprocessed_dir, "y_test_I17.npy"), y_test_I17)
        np.save(os.path.join(preprocessed_dir, "X_train_I18.npy"), X_train_I18)
        np.save(os.path.join(preprocessed_dir, "y_train_I18.npy"), y_train_I18)
        np.save(os.path.join(preprocessed_dir, "X_val_I18.npy"), X_val_I18)
        np.save(os.path.join(preprocessed_dir, "y_val_I18.npy"), y_val_I18)
        np.save(os.path.join(preprocessed_dir, "X_test_I18.npy"), X_test_I18)
        np.save(os.path.join(preprocessed_dir, "y_test_I18.npy"), y_test_I18)

    # Process clean images and add noise
    X_train_I16_processed = preprocess_images(X_train_I16)
    X_train_I17_processed = preprocess_images(X_train_I17)
    X_train_I18_processed = preprocess_images(X_train_I18)
    X_train_PH_processed = preprocess_images(X_train_PH)
    logger.info(f"X_train_PH_processed min: {X_train_PH_processed.min().item()}, max: {X_train_PH_processed.max().item()}")
    logger.info(f"X_train_I16_processed min: {X_train_I16_processed.min().item()}, max: {X_train_I16_processed.max().item()}")
    logger.info(f"X_train_I17_processed min: {X_train_I17_processed.min().item()}, max: {X_train_I17_processed.max().item()}")
    logger.info(f"X_train_I18_processed min: {X_train_I18_processed.min().item()}, max: {X_train_I18_processed.max().item()}")

    x_train_noisy_PH = add_noise(preprocess_images(X_train_PH))
    x_val_noisy_PH = add_noise(preprocess_images(X_val_PH))
    x_test_noisy_PH = add_noise(preprocess_images(X_test_PH))

    x_train_noisy_I16 = add_noise(preprocess_images(X_train_I16))
    x_val_noisy_I16 = add_noise(preprocess_images(X_val_I16))
    x_test_noisy_I16 = add_noise(preprocess_images(X_test_I16))

    x_train_noisy_I17 = add_noise(preprocess_images(X_train_I17))
    x_val_noisy_I17 = add_noise(preprocess_images(X_val_I17))
    x_test_noisy_I17 = add_noise(preprocess_images(X_test_I17))

    x_train_noisy_I18 = add_noise(preprocess_images(X_train_I18))
    x_val_noisy_I18 = add_noise(preprocess_images(X_val_I18))
    x_test_noisy_I18 = add_noise(preprocess_images(X_test_I18))

    # Concatenate datasets (keep as CPU tensors)
    x_train_noisy = torch.cat([x_train_noisy_PH, x_train_noisy_I16, x_train_noisy_I17, x_train_noisy_I18], dim=0)
    x_val_noisy = torch.cat([x_val_noisy_PH, x_val_noisy_I16, x_val_noisy_I17, x_val_noisy_I18], dim=0)
    x_test_noisy = torch.cat([x_test_noisy_PH, x_test_noisy_I16, x_test_noisy_I17, x_test_noisy_I18], dim=0)

    X_train = torch.cat([X_train_PH_processed, X_train_I16_processed, X_train_I17_processed, X_train_I18_processed], dim=0)
    X_val = torch.cat([preprocess_images(X_val_PH), preprocess_images(X_val_I16), preprocess_images(X_val_I17), preprocess_images(X_val_I18)], dim=0)
    X_test = torch.cat([preprocess_images(X_test_PH), preprocess_images(X_test_I16), preprocess_images(X_test_I17), preprocess_images(X_test_I18)], dim=0)

    logger.info(f"X_train min: {X_train.min().item()}, max: {X_train.max().item()}, mean: {X_train.mean().item()}")
    logger.info(f"X_val min: {X_val.min().item()}, max: {X_val.max().item()}, mean: {X_val.mean().item()}")
    logger.info(f"X_test min: {X_test.min().item()}, max: {X_test.max().item()}, mean: {X_test.mean().item()}")

    logger.info(f"x_train_noisy min: {x_train_noisy.min().item()}, max: {x_train_noisy.max().item()}, mean: {x_train_noisy.mean().item()}")
    logger.info(f"x_val_noisy min: {x_val_noisy.min().item()}, max: {x_val_noisy.max().item()}, mean: {x_val_noisy.mean().item()}")
    logger.info(f"x_test_noisy min: {x_test_noisy.min().item()}, max: {x_test_noisy.max().item()}, mean: {x_test_noisy.mean().item()}")

    # Do NOT move to device here
    logger.info("Data preparation complete")
    return (x_train_noisy, X_train), (x_val_noisy, X_val), (x_test_noisy, X_test), (y_train_PH, y_val_PH, y_test_PH)

# Function to count trainable parameters
def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)  # Up to conv4_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output_vgg = self.vgg(output)
        target_vgg = self.vgg(target)
        return self.mse(output_vgg, target_vgg)

# Combined Loss
def combined_loss(output, target, device, alpha=0.7, beta=0.2, gamma=0.1):
    mse_loss = nn.MSELoss()(output, target)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_loss = 1 - ssim_metric(output, target)  # Convert to loss (1 - SSIM)
    perceptual_loss_fn = PerceptualLoss(device)
    perceptual_loss = perceptual_loss_fn(output, target)
    
    return alpha * mse_loss + beta * ssim_loss + gamma * perceptual_loss

# Training function with detailed logging
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, filepath):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for i, (noisy_inputs, clean_targets) in enumerate(train_pbar):
            # Move batches to GPU here
            noisy_inputs = noisy_inputs.to(device)
            clean_targets = clean_targets.to(device)
            
            logger.info(f"Batch {i+1}/{len(train_loader)}: Starting")
            logger.info(f"Batch {i+1}/{len(train_loader)}: Input shape: {noisy_inputs.shape}")
            optimizer.zero_grad()
            logger.info(f"Batch {i+1}/{len(train_loader)}: Forward pass start")
            outputs, _ = model(noisy_inputs)
            logger.info(f"Batch {i+1}/{len(train_loader)}: Forward pass complete, output shape: {outputs.shape}")
            loss = criterion(outputs, clean_targets)
            logger.info(f"Batch {i+1}/{len(train_loader)}: Loss computed: {loss.item()}")
            loss.backward()
            logger.info(f"Batch {i+1}/{len(train_loader)}: Backward pass complete")
            optimizer.step()
            logger.info(f"Batch {i+1}/{len(train_loader)}: Optimizer step complete")
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for i, (noisy_inputs, clean_targets) in enumerate(val_pbar):
                # Move batches to GPU here
                noisy_inputs = noisy_inputs.to(device)
                clean_targets = clean_targets.to(device)
                outputs, _ = model(noisy_inputs)
                loss = criterion(outputs, clean_targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        val_loss /= len(val_loader)

        lr = scheduler(epoch, optimizer.param_groups[0]['lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filepath)
            logger.info(f"Saved best model with Val Loss: {val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    mse = 0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_total, psnr_total = 0, 0
    predictions, embeddings, noisy_inputs_list, clean_targets_list = [], [], [], []

    with torch.no_grad():
        for noisy_inputs, clean_targets in test_loader:
            # Move batches to GPU here
            noisy_inputs = noisy_inputs.to(device)
            clean_targets = clean_targets.to(device)
            outputs, embedding = model(noisy_inputs)
            mse += criterion(outputs, clean_targets).item()
            ssim_total += ssim_metric(outputs, clean_targets).item()
            psnr_total += psnr_metric(outputs, clean_targets).item()
            predictions.append(outputs.cpu())
            embeddings.append(embedding.cpu())
            noisy_inputs_list.append(noisy_inputs.cpu())
            clean_targets_list.append(clean_targets.cpu())

    mse /= len(test_loader)
    ssim = ssim_total / len(test_loader)
    psnr = psnr_total / len(test_loader)

    predictions = torch.cat(predictions, dim=0)
    embeddings = torch.cat(embeddings, dim=0)
    noisy_inputs = torch.cat(noisy_inputs_list, dim=0)
    clean_targets = torch.cat(clean_targets_list, dim=0)

    return mse, ssim, psnr, predictions, embeddings, noisy_inputs, clean_targets

# Plotting function
def save_comparison_plots(noisy_inputs, predictions, clean_targets, output_dir, num_samples=5):
    for i in range(min(num_samples, len(noisy_inputs))):
        logger.info(f"Sample {i} - Noisy min: {noisy_inputs[i].min().item()}, max: {noisy_inputs[i].max().item()}")
        logger.info(f"Sample {i} - Reconstructed min: {predictions[i].min().item()}, max: {predictions[i].max().item()}")
        logger.info(f"Sample {i} - Clean min: {clean_targets[i].min().item()}, max: {clean_targets[i].max().item()}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(noisy_inputs[i].permute(1, 2, 0).numpy())
        axes[0].set_title("Noisy Input")
        axes[0].axis('off')
        axes[1].imshow(predictions[i].permute(1, 2, 0).numpy())
        axes[1].set_title("Reconstructed")
        axes[1].axis('off')
        axes[2].imshow(clean_targets[i].permute(1, 2, 0).numpy())
        axes[2].set_title("Clean Target")
        axes[2].axis('off')
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        plt.close()

def main():
    # Load and prepare data
    (x_train_noisy, X_train), (x_val_noisy, X_val), (x_test_noisy, X_test), _ = prepare_data()

    # Create DataLoaders with pin_memory=True (works with CPU tensors)
    train_dataset = TensorDataset(x_train_noisy, X_train)
    val_dataset = TensorDataset(x_val_noisy, X_val)
    test_dataset = TensorDataset(x_test_noisy, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize model (single GPU for now)
    height, width, channels = size, size, X_train.shape[1]
    auto_enc = AutoencoderMaSA(height, width, channels, "autoencoder_masa", use_masa=True).to(device)
    logger.info("Model initialized on single GPU")

    # Print number of trainable parameters
    num_params = count_trainable_parameters(auto_enc)
    logger.info(f"Number of trainable parameters: {num_params:,}")

    optimizer = optim.Adam(auto_enc.parameters(), lr=learning_rate)
    criterion = criterion = lambda x, y: combined_loss(x, y, device) #nn.MSELoss()

    # Learning rate scheduler
    def scheduler(epoch, lr):
        return lr if epoch < 10 else lr * np.exp(-0.1)

    # Train the model
    filepath = f'models/{dataset_name}_autoencoder.pt'
    auto_enc = train_model(auto_enc, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, filepath)

    # Evaluate model
    auto_enc.eval()
    mse, ssim, psnr, predictions, embeddings, noisy_inputs, clean_targets = evaluate_model(auto_enc, test_loader, criterion)

    # Print metrics
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test SSIM: {ssim:.4f}")
    logger.info(f"Test PSNR: {psnr:.4f}")

    # Save metrics to file
    with open(f'results/{dataset_name}_autoencoder_metrics.txt', 'w') as f:
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test SSIM: {ssim:.4f}\n")
        f.write(f"Test PSNR: {psnr:.4f}\n")
    logger.info(f"Metrics saved to results/{dataset_name}_autoencoder_metrics.txt")

    # Save embeddings
    torch.save(embeddings, f'models/{dataset_name}_embeddings.pt')
    logger.info(f"Saved test embeddings with shape: {embeddings.shape}")

    # Save comparison plots
    save_comparison_plots(noisy_inputs, predictions, clean_targets, "outputs")
    logger.info("Comparison plots saved in outputs/ folder")

if __name__ == "__main__":
    main()
