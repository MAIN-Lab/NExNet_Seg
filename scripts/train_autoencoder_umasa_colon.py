# train_autoencoder.py
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
from torchvision.models import vgg16

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable for CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA calls for better error reporting

# Paths
root_kvasir = "../datasets/kvasir_seg"
root_cvc = "../datasets/CVC_CLINICDB/PNG"
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Hyperparameters
size = 224
dataset_name = "colon"
batch_size = 8  # Reduced to minimize memory issues
num_epochs = 200  # Reduced for debugging
learning_rate = 0.0001  # Reduced to prevent collapse
patience = 20
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
logger.info(f"PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")

# Data loading function
def load_images(image_paths, mask_paths=None):
    images, masks = [], []
    try:
        for img_path in image_paths:
            img = np.array(Image.open(img_path).resize((size, size)), dtype=np.float32) / 255.0
            images.append(img)
        if mask_paths:
            for mask_path in mask_paths:
                mask = np.array(Image.open(mask_path).resize((size, size)), dtype=np.float32) / 255.0
                masks.append(mask[:, :, 0] if mask.ndim == 3 else mask)
            return np.array(images), np.array(masks)
        return np.array(images)
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        raise

# Load and preprocess data
def prepare_data():
    logger.info("Starting data preparation")
    imgs_path_list_tr_kvasir = sorted(glob.glob(os.path.join(root_kvasir, "Train/images", "*.jpg")))
    imgs_path_list_ts_kvasir = sorted(glob.glob(os.path.join(root_kvasir, "Test/images", "*.jpg")))
    masks_path_list_tr_kvasir = sorted(glob.glob(os.path.join(root_kvasir, "Train/masks", "*.jpg")))
    masks_path_list_ts_kvasir = sorted(glob.glob(os.path.join(root_kvasir, "Test/masks", "*.jpg")))

    X_train_kvasir, y_train_kvasir = load_images(imgs_path_list_tr_kvasir, masks_path_list_tr_kvasir)
    X_test_kvasir, y_test_kvasir = load_images(imgs_path_list_ts_kvasir, masks_path_list_ts_kvasir)
    logger.info(f"Kvasir Train: {X_train_kvasir.shape}, Test: {X_test_kvasir.shape}")
    logger.info(f"X_train_kvasir min: {X_train_kvasir.min()}, max: {X_train_kvasir.max()}")

    imgs_path_list_cvc = sorted(glob.glob(os.path.join(root_cvc, "Original", "*.png")))
    masks_path_list_cvc = sorted(glob.glob(os.path.join(root_cvc, "Ground Truth", "*.png")))
    imgs_arr_cvc, masks_arr_cvc = load_images(imgs_path_list_cvc, masks_path_list_cvc)
    X_train_cvc, X_test_cvc, y_train_cvc, y_test_cvc = train_test_split(
        imgs_arr_cvc, masks_arr_cvc, test_size=0.25, random_state=101
    )
    logger.info(f"CVC Train: {X_train_cvc.shape}, Test: {X_test_cvc.shape}")
    logger.info(f"X_train_cvc min: {X_train_cvc.min()}, max: {X_train_cvc.max()}")

    x_rotated_kvasir, y_rotated_kvasir, x_flipped_kvasir, y_flipped_kvasir = img_augmentation(X_train_kvasir, y_train_kvasir)
    X_train_full_kvasir = np.concatenate([X_train_kvasir, x_rotated_kvasir, x_flipped_kvasir])
    y_train_full_kvasir = np.concatenate([y_train_kvasir, y_rotated_kvasir, y_flipped_kvasir])
    X_train_kvasir, X_val_kvasir, y_train_kvasir, y_val_kvasir = train_test_split(
        X_train_full_kvasir, y_train_full_kvasir, test_size=0.20, random_state=101
    )

    x_rotated_cvc, y_rotated_cvc, x_flipped_cvc, y_flipped_cvc = img_augmentation(X_train_cvc, y_train_cvc)
    X_train_full_cvc = np.concatenate([X_train_cvc, x_rotated_cvc, x_flipped_cvc])
    y_train_full_cvc = np.concatenate([y_train_cvc, y_rotated_cvc, y_flipped_cvc])
    X_train_cvc, X_val_cvc, y_train_cvc, y_val_cvc = train_test_split(
        X_train_full_cvc, y_train_full_cvc, test_size=0.20, random_state=101
    )

    # Process clean images with debugging
    X_train_kvasir_processed = preprocess_images(X_train_kvasir)
    logger.info(f"X_train_kvasir_processed min: {X_train_kvasir_processed.min().item()}, max: {X_train_kvasir_processed.max().item()}")
    X_train_cvc_processed = preprocess_images(X_train_cvc)
    logger.info(f"X_train_cvc_processed min: {X_train_cvc_processed.min().item()}, max: {X_train_cvc_processed.max().item()}")

    x_train_noisy_kvasir = add_noise(preprocess_images(X_train_kvasir))
    x_val_noisy_kvasir = add_noise(preprocess_images(X_val_kvasir))
    x_test_noisy_kvasir = add_noise(preprocess_images(X_test_kvasir))

    x_train_noisy_cvc = add_noise(preprocess_images(X_train_cvc))
    x_val_noisy_cvc = add_noise(preprocess_images(X_val_cvc))
    x_test_noisy_cvc = add_noise(preprocess_images(X_test_cvc))

    x_train_noisy = torch.cat([x_train_noisy_kvasir, x_train_noisy_cvc], dim=0)
    x_val_noisy = torch.cat([x_val_noisy_kvasir, x_val_noisy_cvc], dim=0)
    x_test_noisy = torch.cat([x_test_noisy_kvasir, x_test_noisy_cvc], dim=0)

    X_train = torch.cat([X_train_kvasir_processed, X_train_cvc_processed], dim=0)
    X_val = torch.cat([preprocess_images(X_val_kvasir), preprocess_images(X_val_cvc)], dim=0)
    X_test = torch.cat([preprocess_images(X_test_kvasir), preprocess_images(X_test_cvc)], dim=0)

    logger.info(f"X_train min: {X_train.min().item()}, max: {X_train.max().item()}, mean: {X_train.mean().item()}")
    logger.info(f"X_val min: {X_val.min().item()}, max: {X_val.max().item()}, mean: {X_val.mean().item()}")
    logger.info(f"X_test min: {X_test.min().item()}, max: {X_test.max().item()}, mean: {X_test.mean().item()}")

    logger.info(f"x_train_noisy min: {x_train_noisy.min().item()}, max: {x_train_noisy.max().item()}, mean: {x_train_noisy.mean().item()}")
    logger.info(f"x_val_noisy min: {x_val_noisy.min().item()}, max: {x_val_noisy.max().item()}, mean: {x_val_noisy.mean().item()}")
    logger.info(f"x_test_noisy min: {x_test_noisy.min().item()}, max: {x_test_noisy.max().item()}, mean: {x_test_noisy.mean().item()}")

    # Remove the incorrect permute, already in (N, C, H, W) format
    x_train_noisy = x_train_noisy.to(device)
    x_val_noisy = x_val_noisy.to(device)
    x_test_noisy = x_test_noisy.to(device)
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    logger.info("Data preparation complete")
    return (x_train_noisy, X_train), (x_val_noisy, X_val), (x_test_noisy, X_test), (y_train_kvasir, y_val_kvasir, y_test_kvasir)

# Function to count trainable parameters
def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# Training function with detailed logging
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, filepath):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for i, (noisy_inputs, clean_targets) in enumerate(train_pbar):
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
            torch.save(model.state_dict(), filepath)  # Simplified for single GPU
            logger.info(f"Saved best model with Val Loss: {val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model

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

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_noisy, X_train)
    val_dataset = TensorDataset(x_val_noisy, X_val)
    test_dataset = TensorDataset(x_test_noisy, X_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # Initialize model (single GPU for now)
    height, width, channels = size, size, X_train.shape[1]
    auto_enc = AutoencoderMaSA(height, width, channels, "autoencoder_masa", use_masa=True).to(device)
    logger.info("Model initialized on single GPU")

    # Print number of trainable parameters
    num_params = count_trainable_parameters(auto_enc)
    logger.info(f"Number of trainable parameters: {num_params:,}")

    optimizer = optim.Adam(auto_enc.parameters(), lr=learning_rate)
    criterion = criterion = lambda x, y: combined_loss(x, y, device)# nn.MSELoss()

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
