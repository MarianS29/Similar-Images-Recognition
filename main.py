import os
import random
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classes import ContrastiveLoss, SiameseDataset, SiameseNet
from utils import (
    compute_embeddings,
    eval_one_epoch,
    set_seed,
    show_similar,
    train_one_epoch,
)

# ============================================================
# 1. Configuration & Setup
# ============================================================


@dataclass
class Config:
    """Hyperparameters and configuration settings."""

    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 32
    LR: float = 1e-4
    EPOCHS: int = 5
    EMBEDDING_DIM: int = 128
    TEST_SIZE: float = 0.2
    Dataset_ID: str = "theaayushbajaj/cbir-dataset"
    OUTPUT_EMB_FILE: str = os.path.join("models", "cbir_embeddings.pt")
    OUTPUT_CSV_FILE: str = os.path.join("models", "cbir_image_paths.csv")


def get_transforms():
    """Returns training and validation transformations."""
    train_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


# ============================================================
# 2. Data Preparation
# ============================================================


def prepare_data(config: Config):
    """Downloads dataset, parses files, and splits into train/val."""
    print("Downloading/Loading dataset...")
    path = kagglehub.dataset_download(config.Dataset_ID)
    data_root = Path(path)
    img_root = data_root / "dataset"

    print(f"Data root: {data_root}")

    # Find all jpg images
    image_paths = sorted(img_root.glob("*.jpg"))
    print(f"Total images found: {len(image_paths)}")

    if not image_paths:
        raise FileNotFoundError("No .jpg files found. Check dataset structure.")

    # Create master DataFrame
    df = pd.DataFrame({"path": image_paths})

    # Split
    train_df, val_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    return train_df, val_df


# ============================================================
# 3. Training Loop
# ============================================================


def run_training(model, train_loader, val_loader, optimizer, criterion, config):
    """Executes the training loop over the specified epochs."""
    print("\nStarting training...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        val_loss = eval_one_epoch(model, val_loader, criterion, config.DEVICE)

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    return model


# ============================================================
# 4. Main Execution
# ============================================================


def main():
    # A. Setup
    cfg = Config()
    set_seed(cfg.SEED)
    print(f"Running on device: {cfg.DEVICE}")

    # B. Data Preparation
    train_df, val_df = prepare_data(cfg)
    train_tf, val_tf = get_transforms()

    train_dataset = SiameseDataset(train_df, transform=train_tf)
    val_dataset = SiameseDataset(val_df, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # C. Model Initialization
    model = SiameseNet(embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    # D. Training
    model = run_training(model, train_loader, val_loader, optimizer, criterion, cfg)

    # E. Compute Embeddings for Entire Database
    print("\nComputing embeddings for the full dataset...")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    db_embeddings = compute_embeddings(
        model, full_df, val_tf, cfg.DEVICE, batch_size=64
    )
    print(f"Embeddings shape: {db_embeddings.shape}")

    # F. Save Artifacts
    torch.save(db_embeddings, cfg.OUTPUT_EMB_FILE)
    full_df.to_csv(cfg.OUTPUT_CSV_FILE, index=False)
    print(f"Saved embeddings to {cfg.OUTPUT_EMB_FILE}")

    # G. Visualization / Test
    #  - Placeholder for concept
    print("\nVisualizing random query result...")
    random_idx = random.randint(0, len(full_df) - 1)

    show_similar(
        query_idx=random_idx,
        full_df=full_df,
        model=model,
        db_embeddings=db_embeddings,
        transform=val_tf,
        device=cfg.DEVICE,
    )


if __name__ == "__main__":
    main()
