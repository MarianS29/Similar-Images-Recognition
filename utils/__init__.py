import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def compute_embeddings(model, df, transform, device, batch_size=64):
    """
    Computes embeddings for all images in the provided DataFrame.

    Args:
        model: The trained Siamese network.
        df: DataFrame containing a 'path' column.
        transform: Transformations to apply to images (usually val_transform).
        device: 'cuda' or 'cpu'.
        batch_size: Batch size for processing.

    Returns:
        Tensor of shape [N, Embedding_Dim]
    """
    model.eval()
    all_embeddings = []

    for start in range(0, len(df), batch_size):
        batch_paths = df["path"].iloc[start : start + batch_size].tolist()
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        imgs = torch.stack(imgs).to(device)

        # Assuming model has a sub-network 'embedding_net' or 'cnn'+'fc'
        # If your model's forward_once is mapped to model.embedding_net:
        if hasattr(model, "embedding_net"):
            z = model.embedding_net(imgs)
        else:
            # Fallback if the user didn't wrap layers in 'embedding_net'
            # Adjust this based on your specific SiameseNet implementation
            z = model.forward_once(imgs)

        all_embeddings.append(z.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    return all_embeddings


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """
    Evaluates the Siamese network on the validation set.
    """
    model.eval()
    running_loss = 0.0

    for img1, img2, target in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        z1, z2 = model(img1, img2)
        loss = criterion(z1, z2, target)
        running_loss += loss.item()

    return running_loss / len(loader)


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def show_similar(query_idx, full_df, model, db_embeddings, transform, device, top_k=5):
    """
    Visualizes the query image and the top_k most similar images from the database.

    Args:
        query_idx: Index of the query image in full_df.
        full_df: DataFrame containing image paths.
        model: Trained Siamese model.
        db_embeddings: Precomputed embeddings for full_df (Tensor).
        transform: Validation transform for the query image.
        device: 'cuda' or 'cpu'.
        top_k: Number of similar images to show.
    """
    model.eval()

    query_path = full_df.loc[query_idx, "path"]
    query_img = Image.open(query_path).convert("RGB")

    # Embedding for the query image
    query_tensor = transform(query_img).unsqueeze(0).to(device)

    # Handle model structure (embedding_net vs forward_once)
    if hasattr(model, "embedding_net"):
        query_emb = model.embedding_net(query_tensor).cpu()
    else:
        query_emb = model.forward_once(query_tensor).cpu()

    # Calculate L2 distances against the entire database
    dists = torch.cdist(query_emb, db_embeddings, p=2).squeeze(0)  # [N]

    # Sort ascending (smaller distance = more similar)
    sorted_indices = torch.argsort(dists)

    # Plotting
    plt.figure(figsize=(3 * (top_k + 1), 4))

    # Show Query Image
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.axis("off")
    plt.title("Query")

    col = 2
    for idx in sorted_indices:
        idx = idx.item()
        # Skip the exact same image if it appears (usually rank 0)
        if idx == query_idx:
            continue

        res_path = full_df.loc[idx, "path"]
        res_img = Image.open(res_path).convert("RGB")

        plt.subplot(1, top_k + 1, col)
        plt.imshow(res_img)
        plt.axis("off")
        plt.title(f"Rank {col - 1}\nDist: {dists[idx]:.4f}")
        col += 1

        if col > top_k + 1:
            break

    plt.tight_layout()
    plt.show()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the Siamese network for one epoch.
    """
    model.train()
    running_loss = 0.0

    for img1, img2, target in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        z1, z2 = model(img1, img2)
        loss = criterion(z1, z2, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)
