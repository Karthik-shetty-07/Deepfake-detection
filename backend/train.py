"""
Training Script for Deepfake Detection Model.

Fine-tunes the XceptionNet CNN on your custom dataset of real and deepfake videos.

Usage:
    python train.py --data_dir ../data --epochs 10 --batch_size 8
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
from models.cnn_detector import XceptionNet


class FaceDataset(Dataset):
    """Dataset of face images extracted from videos."""
    
    def __init__(self, face_paths: List[Tuple[str, int]], transform=None):
        """
        Args:
            face_paths: List of (image_path, label) where label is 0=real, 1=fake
            transform: Optional transforms
        """
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
    
    Returns:
        Dictionary with 'real' and 'deepfake' keys containing face image paths
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
        
        # Get all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        videos = [f for f in category_path.iterdir() 
                  if f.suffix.lower() in video_extensions]
        
        if not videos:
            print(f"[Warning] No videos found in {category_path}")
            continue
        
        print(f"\n[Training] Processing {len(videos)} {category} videos...")
        
        # Create output directory for this category
        cat_output = output_path / category
        cat_output.mkdir(exist_ok=True)
        
        for video_file in tqdm(videos, desc=f"Extracting {category}"):
            try:
                # Extract faces
                faces = processor.process_video(str(video_file))
                
                # Save face images
                for i, face_data in enumerate(faces):
                    face_img = Image.fromarray(face_data.face_image[:, :, ::-1])  # BGR to RGB
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
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Combine paths with labels
    all_data = []
    for path in face_paths['real']:
        all_data.append((path, 0))  # 0 = real
    for path in face_paths['deepfake']:
        all_data.append((path, 1))  # 1 = fake
    
    # Shuffle
    random.shuffle(all_data)
    
    # Split
    val_size = int(len(all_data) * val_split)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    
    print(f"[Training] Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = FaceDataset(train_data, transform=train_transform)
    val_dataset = FaceDataset(val_data, transform=val_transform)
    
    # Create loaders
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
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
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
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='weights/trained_model.pth',
                        help='Output path for trained model')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using device: {device}")
    
    # Initialize video processor
    processor = VideoProcessor(frames_to_extract=16, device=str(device))
    
    # Extract faces from videos
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
    
    train_loader, val_loader = create_data_loaders(
        face_paths, batch_size=args.batch_size
    )
    
    # Initialize model
    print("\n" + "="*50)
    print("Step 3: Initializing model...")
    print("="*50)
    
    model = XceptionNet(pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    print("\n" + "="*50)
    print("Step 4: Training...")
    print("="*50)
    
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
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
                'train_date': datetime.now().isoformat()
            }, output_path)
            print(f"✓ Saved best model with {best_acc*100:.2f}% accuracy")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {args.output}")
    print("\nTo use the trained model, the system will automatically")
    print("load it from the weights/ directory on next startup.")


if __name__ == '__main__':
    main()
