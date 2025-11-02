import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import logging

# Add project root to path for edgeface import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.dataset import TripletFaceDataset
from edgeface.backbones import get_model
from config import (
    EDGEFACE_CHECKPOINT_DIR, EDGEFACE_MODEL_NAME, MODELS_DIR,
    DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, BATCH_SIZE, TRIPLET_MARGIN,
    MODEL_INPUT_SIZE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(dataset_path: str, model_name: str, num_epochs: int = None, learning_rate: float = None):
    """
    Train face recognition model using triplet loss

    Args:
        dataset_path: Path to dataset directory
        model_name: Name for saved model
        num_epochs: Number of training epochs (default: from config)
        learning_rate: Learning rate (default: from config)
    """
    # Use defaults from config if not specified
    num_epochs = num_epochs or DEFAULT_EPOCHS
    learning_rate = learning_rate or DEFAULT_LEARNING_RATE

    logger.info(f"Starting training: {num_epochs} epochs, lr={learning_rate}")

    # --- 1. Setup Device ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    # --- 2. Load Pre-trained Model ---
    try:
        model = get_model(EDGEFACE_MODEL_NAME)
        checkpoint_path = os.path.join(EDGEFACE_CHECKPOINT_DIR, f"{EDGEFACE_MODEL_NAME}.pt")

        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
            logger.info(f"Loaded pre-trained model from {checkpoint_path}")
        else:
            logger.warning(f"Pre-trained model not found at {checkpoint_path}, training from scratch")

        model.to(DEVICE)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # --- 3. Prepare DataLoader ---
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        train_dataset = TripletFaceDataset(root_dir=dataset_path, transform=train_transforms)

        if len(train_dataset) == 0:
            logger.error("Dataset is empty! Please add training images.")
            raise ValueError("Empty dataset")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        logger.info(f"Dataset loaded: {len(train_dataset)} triplets")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # --- 4. Setup Training Components ---
    loss_function = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # --- 5. Training Loop ---
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0

        for i, (anchor, positive, negative) in enumerate(train_loader):
            try:
                anchor = anchor.to(DEVICE)
                positive = positive.to(DEVICE)
                negative = negative.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)

                # Calculate loss
                loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)

                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                # Log progress
                if (i + 1) % 10 == 0:
                    avg_loss = running_loss / batch_count
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")

            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                continue

        # Epoch summary
        epoch_loss = running_loss / max(batch_count, 1)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")

        # Update learning rate
        scheduler.step(epoch_loss)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path}")

    logger.info("Training finished!")

    # --- 6. Save Final Model ---
    final_save_path = os.path.join(MODELS_DIR, f"{model_name}_final.pt")
    torch.save(model.state_dict(), final_save_path)
    logger.info(f"Final model saved to {final_save_path}")

    return final_save_path


if __name__ == '__main__':
    # Example usage
    train_model("dataset", "my_model")
