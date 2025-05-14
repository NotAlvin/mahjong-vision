import os
import argparse
import random
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from yolo_detector import MahjongTileDetector

class MahjongTileClassifier:
    """Class for classifying mahjong tiles using a CNN."""
    
    # Map of class indices to mahjong tile names
    TILE_CLASSES = {
        0: "bamboo_1", 1: "bamboo_2", 2: "bamboo_3", 3: "bamboo_4", 
        4: "bamboo_5", 5: "bamboo_6", 6: "bamboo_7", 7: "bamboo_8", 8: "bamboo_9",
        9: "character_1", 10: "character_2", 11: "character_3", 12: "character_4",
        13: "character_5", 14: "character_6", 15: "character_7", 16: "character_8", 17: "character_9",
        18: "circle_1", 19: "circle_2", 20: "circle_3", 21: "circle_4", 
        22: "circle_5", 23: "circle_6", 24: "circle_7", 25: "circle_8", 26: "circle_9",
        27: "east", 28: "green", 29: "north", 30: "red", 31: "south", 32: "west", 33: "white",
        # Additional classes for bonus tiles if needed
    }

    def __init__(self, model_path=None, num_classes=34, input_size=224):
        """
        Initialize the classifier model.
        
        Args:
            model_path: Path to a pre-trained model. If None, creates a new one.
            num_classes: Number of mahjong tile classes
            input_size: Size of the input image (will be square)
        """
        self.input_size = input_size
        # Set up image transformations
        # Transform that maintains aspect ratio for inference
        self.transform = transforms.Compose([
            # Resize the shortest side to input_size while maintaining aspect ratio
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # Pad to make square if needed
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create model
        self.model = self._create_model(num_classes)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _create_model(self, num_classes):
        """
        Create a model for tile classification.
        
        Args:
            num_classes: Number of mahjong tile classes
            
        Returns:
            PyTorch model
        """
        # Using EfficientNet which handles various aspect ratios better
        model = models.efficientnet_b0(weights=None)
        
        # Replace the final classification layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        return model
    
    def classify_tile(self, tile_image, use_tta=True):
        """
        Classify a single mahjong tile with optional test-time augmentation.
        
        Args:
            tile_image: OpenCV image of a mahjong tile
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Predicted class name and confidence score
        """
        # Convert BGR to RGB
        tile_rgb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(tile_rgb)
        
        # Define base transform
        base_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if use_tta:
            # Define test-time augmentations
            tta_transforms = [
                base_transform,
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.0),
                    *base_transform.transforms
                ]),
                transforms.Compose([
                    transforms.RandomRotation(15, expand=True),
                    *base_transform.transforms
                ]),
                transforms.Compose([
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    *base_transform.transforms
                ]),
            ]
            
            # Get predictions for each augmentation
            predictions = []
            for t in tta_transforms:
                input_tensor = t(pil_image).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predictions.append(torch.softmax(output, dim=1))
            
            # Average predictions
            avg_prediction = torch.mean(torch.stack(predictions), dim=0)
            confidence, predicted_idx = torch.max(avg_prediction, 1)
        else:
            # Standard single prediction
            input_tensor = base_transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
        
        # predicted_class and confidence_score are already set by TTA or single prediction
        predicted_class = predicted_idx.item()
        confidence_score = confidence.item()
        
        # Get class name
        class_name = self.TILE_CLASSES.get(predicted_class, f"Unknown ({predicted_class})")
        
        return predicted_class, confidence_score
    
    class MahjongTileDataset(Dataset):        
        def __init__(self, data_dir, transform=None):
            """
            Custom dataset for mahjong tile classification.
            
            Args:
                data_dir: Directory containing 'images' and 'labels' subdirectories
                transform: Optional transform to be applied on a sample
            """
            self.data_dir = Path(data_dir)
            self.image_dir = self.data_dir / 'images'
            self.label_dir = self.data_dir / 'labels'
            # Default transform that maintains aspect ratio
            target_size = 224  # Standard size for ImageNet models
            self.transform = transform or transforms.Compose([
                # Resize the shortest side to target_size while maintaining aspect ratio
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
                # Pad to make square if needed
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Get list of image files (remove extension for matching with labels)
            self.image_files = [f.stem for f in self.image_dir.glob('*.jpg')]  # Adjust extension if needed
            
            # Verify that each image has a corresponding label file
            self.valid_indices = []
            for i, img_stem in enumerate(self.image_files):
                label_path = self.label_dir / f"{img_stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        if f.read().strip():  # Check if file is not empty
                            self.valid_indices.append(i)
            
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            # Get the actual index from valid_indices
            actual_idx = self.valid_indices[idx]
            img_name = self.image_files[actual_idx]
            img_path = self.image_dir / f"{img_name}.jpg"
            
            # Load the full image
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            # Get corresponding label file
            label_path = self.label_dir / f"{img_name}.txt"
            
            # Read all ground truth boxes and classes
            boxes = []
            class_ids = []
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 9:  # class_id + 8 points (x1,y1,x2,y2,x3,y3,x4,y4)
                        print(f"Warning: Invalid line format in {label_path}: {line}")
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        # Convert polygon points to list of floats
                        points = [float(x) for x in parts[1:]]
                        
                        # Convert from normalized [0,1] to pixel coordinates
                        points[0::2] = [x * img_width for x in points[0::2]]  # x coordinates
                        points[1::2] = [y * img_height for y in points[1::2]]  # y coordinates
                        
                        # Get bounding box that contains all points
                        x_coords = points[0::2]  # All x coordinates
                        y_coords = points[1::2]  # All y coordinates
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                        
                        # Add some padding (5% of width/height)
                        w, h = x2 - x1, y2 - y1
                        x1 = max(0, x1 - 0.05 * w)
                        y1 = max(0, y1 - 0.05 * h)
                        x2 = min(img_width, x2 + 0.05 * w)
                        y2 = min(img_height, y2 + 0.05 * h)
                        
                        # Ensure we have a valid box
                        if x1 < x2 and y1 < y2:
                            boxes.append((x1, y1, x2, y2))
                            class_ids.append(class_id)
                            
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line in {label_path}: {line} - {e}")
            
            if not boxes:
                # If no valid boxes found, use the whole image
                box = (0, 0, img_width, img_height)
                class_id = 0  # Default class if no class found
            else:
                # For now, just use the first box if multiple exist
                box = boxes[0]
                class_id = class_ids[0]
            
            # Ensure the bounding box is valid (has positive area)
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure the crop has positive dimensions
            if x1 >= x2 or y1 >= y2:
                # If invalid box, use a small center crop of the image
                min_size = min(img_width, img_height)
                crop_size = max(32, min_size // 4)  # At least 32x32 pixels
                center_x, center_y = img_width // 2, img_height // 2
                half_size = crop_size // 2
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img_width, center_x + half_size)
                y2 = min(img_height, center_y + half_size)
                print(f"Warning: Invalid bounding box in {img_path}, using center crop")
            
            # Ensure we have at least some area to crop
            if x1 >= x2 or y1 >= y2:
                # If still invalid, use the whole image
                x1, y1, x2, y2 = 0, 0, img_width, img_height
                print(f"Warning: Using full image for {img_path} due to invalid box")
            
            try:
                cropped_img = image.crop((x1, y1, x2, y2))
                
                # Double-check the crop is valid
                if cropped_img.width == 0 or cropped_img.height == 0:
                    raise ValueError("Zero-sized crop")
                
                # Apply transformations
                if self.transform:
                    cropped_img = self.transform(cropped_img)
                    
                return cropped_img, class_id
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Return a black image of expected size as fallback
                # Use the same target size as in the transform (224 by default)
                target_size = 224  # Default if not specified in transform
                if hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if isinstance(t, transforms.Resize):
                            if isinstance(t.size, int):
                                target_size = t.size
                            elif isinstance(t.size, (tuple, list)) and len(t.size) > 0:
                                target_size = t.size[0]
                            break
                
                fallback_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
                if self.transform:
                    fallback_img = self.transform(fallback_img)
                return fallback_img, class_id

    def train(self, data_root, batch_size=64, num_epochs=30, learning_rate=0.002, val_split=0.1):
        """
        Train the model on a dataset of mahjong tiles.
        
        Args:
            data_root: Root directory containing 'train', 'valid', and 'test' subdirectories
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            val_split: Fraction of training data to use for validation if validation set is not provided
        """
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Define data transformations with more aggressive augmentations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45, expand=True),  # Increased rotation range
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # More conservative validation transforms
        val_transform = transforms.Compose([
            transforms.Resize((int(self.input_size * 1.2), int(self.input_size * 1.2))),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dir = Path(data_root) / 'train'
        val_dir = Path(data_root) / 'valid'
        
        # Check if validation set exists, otherwise split training set
        if val_dir.exists() and any(val_dir.iterdir()):
            train_dataset = self.MahjongTileDataset(train_dir, transform=train_transform)
            val_dataset = self.MahjongTileDataset(val_dir, transform=val_transform)
            print(f"Using separate validation set with {len(val_dataset)} samples")
        else:
            print("No separate validation set found, splitting training set...")
            full_dataset = self.MahjongTileDataset(train_dir, transform=train_transform)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            print(f"Split training set into {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, pin_memory=False)
        
        # Initialize model
        self.model = self.model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': 100 * correct / total
                })
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'loss': val_loss / total,
                        'acc': 100 * correct / total
                    })
            
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = 100 * correct / total
            val_losses.append(val_epoch_loss)
            val_accs.append(val_epoch_acc)
            
            # Step the learning rate scheduler
            scheduler.step(val_epoch_loss)
            
            # Save best model
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"\nSaved new best model with validation loss: {best_val_loss:.4f}")
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%\n')
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, train_accs, val_accs)
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training complete! Best model saved as 'best_model.pth'")
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training and validation metrics."""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


class MahjongRecognitionSystem:
    """Complete system for mahjong tile recognition."""
    
    def __init__(self, detector_model=None, classifier_model=None):
        """
        Initialize the system with detector and classifier models.
        
        Args:
            detector_model: Path to YOLO model for detection
            classifier_model: Path to CNN model for classification
        """
        self.detector = MahjongTileDetector(model_path=detector_model)
        self.classifier = MahjongTileClassifier(model_path=classifier_model)
    
    def process_image(self, image_path, conf_threshold=0.25, visualize=True):
        """
        Process an image to detect and classify mahjong tiles.
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold for detection
            visualize: Whether to visualize the results
            
        Returns:
            tuple: (results, annotated_image) where:
                - results: List of dictionaries with detection and classification info
                - annotated_image: Image with bounding boxes and labels
        """
        # Detect tiles
        cropped_tiles, detections, original_img = self.detector.detect_tiles(
            image_path, conf_threshold)
        
        results = []
        img_with_labels = original_img.copy()
        
        # Classify each tile
        for i, (crop, detection) in enumerate(zip(cropped_tiles, detections)):
            # Get detection info
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Classify the tile
            class_name, confidence = self.classifier.classify_tile(crop)
            
            # Store results
            results.append({
                'tile_id': i,
                'bbox': bbox,
                'class': class_name,
                'confidence': float(confidence),
                'detection_confidence': detection['confidence'],
                'detection_class': detection['class_name']
            })
            
            # Add label to image
            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_labels, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Visualize results
        if visualize:
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(img_with_labels, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Mahjong Tiles Recognition Results')
            plt.show()
        
        return results, img_with_labels
        
    def classify_single_tile(self, tile_image, detection_confidence=1.0, detection_class='tile'):
        """
        Classify a single cropped tile image.
        
        Args:
            tile_image: OpenCV image of a single mahjong tile
            detection_confidence: Confidence score from the detector (default: 1.0)
            detection_class: Class name from the detector (default: 'tile')
            
        Returns:
            dict: Classification result with class name and confidence
        """
        # Classify the tile
        class_name, confidence = self.classifier.classify_tile(tile_image)
        
        return {
            'class': class_name,
            'confidence': float(confidence),
            'detection_confidence': float(detection_confidence),
            'detection_class': detection_class
        }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test Mahjong Tile Classifier')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='datasets',
                      help='Root directory containing train/valid/test folders')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size per GPU/CPU for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'efficientnet_b0'],
                      help='Base model architecture')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pre-trained weights')
    parser.add_argument('--input_size', type=int, default=224,
                      help='Input image size')
    
    # Training options
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                      help='Number of epochs to wait before early stopping')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    # Augmentation options
    parser.add_argument('--color_jitter', type=float, default=0.4,
                      help='Color jitter factor (0-1)')
    parser.add_argument('--auto_augment', action='store_true',
                      help='Use AutoAugment policy')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10,
                      help='Log every N batches')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=1,
                      help='Run validation every N epochs')
    parser.add_argument('--test_image', type=str, default=None,
                      help='Path to a test image for inference (optional)')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize the classifier with the specified model
    print(f"Initializing Mahjong Tile Classifier with {args.model_name}...")
    classifier = MahjongTileClassifier(
        model_path=args.resume,
        num_classes=34,  # Number of mahjong tile classes
        input_size=args.input_size,
    )
    
    # Train the model
    print("\n=== Starting Training ===")
    print(f"Using mixed precision: {args.mixed_precision}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    
    classifier.train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    
    # Initialize the recognition system
    recognition_system = MahjongRecognitionSystem()
    
    # Test the model on a random image from the test set if no specific image is provided
    if not args.test_image:
        test_images = glob.glob(os.path.join(args.data_root, 'test', 'images', '*.jpg'))
        if not test_images:
            print("\nNo test images found in the test directory.")
            return
        args.test_image = random.choice(test_images)
    
    print(f"\n=== Testing on {args.test_image} ===")
    
    # Process the image (detect and classify tiles)
    results, annotated_image = recognition_system.process_image(
        args.test_image,
        conf_threshold=0.25,
        visualize=False  # We'll handle visualization ourselves
    )
    
    # Convert BGR to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis('off')
    
    # Print results to console
    print("\nDetected Tiles:")
    for i, result in enumerate(results, 1):
        print(f"Tile {i}: {result['class']} (Confidence: {result['confidence']:.2f})")
    
    # Save the result
    output_path = 'classification_result.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved classification result to {output_path}")
    
    # If this is a test set image, show the ground truth
    if 'test' in args.test_image:
        label_path = args.test_image.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            print("\nGround Truth:")
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        gt_class = classifier.TILE_CLASSES.get(class_id, f'Unknown ({class_id})')
                        print(f"- {gt_class}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
