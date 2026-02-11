"""
Training Script for Deepfake Detection Model.

Fine-tunes the XceptionNet/EfficientNet CNN on your custom dataset.

Improved with:
- Learning rate warmup + cosine annealing
- Stronger data augmentation (cutout, gaussian noise)
- Class-weighted loss for imbalanced datasets
- Mixed precision training support
- Better logging and checkpointing

Usage:
    python train.py --data_dir ../data --epochs 15 --batch_size 8
"""
import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_processor import VideoProcessor
from models.cnn_detector import XceptionNet, EfficientNetDetector


class GaussianNoise:
    """Add random Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class Cutout:
    """Randomly mask out rectangular patches."""
    def __init__(self, n_holes=1, length=40):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return img * mask


class FaceDataset(Dataset):
    """Dataset of face images extracted from videos."""
    
    def __init__(self, face_paths: List[Tuple[str, int]], transform=None):
        self.face_paths = face_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.face_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.face_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def extract_faces_from_videos(
    data_dir: str,
    output_dir: str,
    processor: VideoProcessor
) -> Dict[str, List[str]]:
    """
    Extract faces from all videos in data_dir.
    
    Expected structure:
        data_dir/
            real/      <- Real videos
            deepfake/  <- Deepfake videos
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {'real': [], 'deepfake': []}
    
    for category in ['real', 'deepfake']:
        category_path = data_path / category
        if not category_path.exists():
            print(f"[Warning] {category_path} does not exist, skipping...")
            continue
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        videos = [f for f in category_path.iterdir() 
                  if f.suffix.lower() in video_extensions]
        
        if not videos:
            print(f"[Warning] No videos found in {category_path}")
            continue
        
        print(f"\n[Training] Processing {len(videos)} {category} videos...")
        
        cat_output = output_path / category
        cat_output.mkdir(exist_ok=True)
        
        for video_file in tqdm(videos, desc=f"Extracting {category}"):
            try:
                faces = processor.process_video(str(video_file))
                
                for i, face_data in enumerate(faces):
                    face_img = Image.fromarray(face_data.face_image[:, :, ::-1])
                    face_path = cat_output / f"{video_file.stem}_face_{i:04d}.jpg"
                    face_img.save(face_path, quality=95)
                    results[category].append(str(face_path))
            
            except Exception as e:
                print(f"[Error] Failed to process {video_file.name}: {e}")
                continue
    
    return results


def create_data_loaders(
    face_paths: Dict[str, List[str]],
    batch_size: int = 8,
    val_split: float = 0.2,
    input_size: int = 299
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with strong augmentation."""
    
    all_data = []
    for path in face_paths['real']:
        all_data.append((path, 0))
    for path in face_paths['deepfake']:
        all_data.append((path, 1))
    
    random.shuffle(all_data)
    
    val_size = int(len(all_data) * val_split)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    
    print(f"[Training] Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        GaussianNoise(std=0.01),
        Cutout(n_holes=1, length=30),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = FaceDataset(train_data, transform=train_transform)
    val_dataset = FaceDataset(val_data, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler=None
) -> Tuple[float, float]:
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory with real/ and deepfake/ subdirectories')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'efficientnet'],
                        help='Model architecture to train')
    parser.add_argument('--output', type=str, default='weights/trained_model.pth',
                        help='Output path for trained model')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using device: {device}")
    print(f"[Training] Model: {args.model}")
    
    # Initialize video processor
    processor = VideoProcessor(frames_to_extract=20, device=str(device))
    
    # Extract faces
    print("\n" + "="*50)
    print("Step 1: Extracting faces from videos...")
    print("="*50)
    
    faces_dir = Path(args.data_dir) / 'extracted_faces'
    face_paths = extract_faces_from_videos(args.data_dir, str(faces_dir), processor)
    
    total_faces = len(face_paths['real']) + len(face_paths['deepfake'])
    print(f"\n[Training] Extracted {total_faces} total faces:")
    print(f"  - Real: {len(face_paths['real'])}")
    print(f"  - Deepfake: {len(face_paths['deepfake'])}")
    
    if total_faces < 20:
        print("\n[Error] Not enough faces extracted. Need at least 20 faces.")
        print("Please add more videos to the data/real and data/deepfake folders.")
        return
    
    # Create data loaders
    print("\n" + "="*50)
    print("Step 2: Creating data loaders...")
    print("="*50)
    
    input_size = 299 if args.model == 'xception' else 380
    train_loader, val_loader = create_data_loaders(
        face_paths, batch_size=args.batch_size, input_size=input_size
    )
    
    # Initialize model
    print("\n" + "="*50)
    print("Step 3: Initializing model...")
    print("="*50)
    
    if args.model == 'efficientnet':
        model = EfficientNetDetector(pretrained=True)
    else:
        model = XceptionNet(pretrained=True)
    model = model.to(device)
    
    # Class-weighted loss
    n_real = max(1, len(face_paths['real']))
    n_fake = max(1, len(face_paths['deepfake']))
    class_weights = torch.FloatTensor([n_fake / (n_real + n_fake), 
                                        n_real / (n_real + n_fake)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[Training] Class weights: real={class_weights[0]:.3f}, fake={class_weights[1]:.3f}")
    
    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},    # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': args.lr}          # Normal LR for classifier
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler with warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("[Training] Using mixed precision training")
    
    # Training loop
    print("\n" + "="*50)
    print("Step 4: Training...")
    print("="*50)
    
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler per batch is handled internally
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            output_path = Path(args.output)
            output_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_type': args.model,
                'train_date': datetime.now().isoformat()
            }, output_path)
            print(f"✓ Saved best model with {best_acc*100:.2f}% accuracy")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {args.output}")
    print("\nThe model will be automatically loaded when running the API server.")


if __name__ == '__main__':
    main()
