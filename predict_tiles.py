import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from yolo_detector import MahjongTileDetector
from tile_classifier import MahjongTileClassifier

def main(image_path):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and classify mahjong tiles in an image')
    parser.add_argument('--image_path', type=str, help='Path to the input image', default=image_path)
    parser.add_argument('--detector_weights', type=str, default='runs/detect/mahjong_detector7/weights/best.pt',
                       help='Path to YOLO detector weights')
    parser.add_argument('--classifier_weights', type=str, default='best_model.pth',
                       help='Path to CNN classifier weights')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output images')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detection')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    detector = MahjongTileDetector(args.detector_weights)
    classifier = MahjongTileClassifier(args.classifier_weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.model = classifier.model.to(device)
    print(f"Using device: {device}")
    
    # Read and check the image
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image: {args.image_path}")
    
    # Run detection
    print("Detecting tiles...")
    cropped_tiles, detections, _ = detector.detect_tiles(args.image_path, conf_threshold=args.conf_threshold)
    
    if not detections:
        print("No tiles detected in the image.")
        return
    
    print(f"Detected {len(detections)} tiles. Classifying...")
    
    # Create a copy of the image for drawing
    output_image = image.copy()
    
    # Process each detection
    for i, (tile, det) in enumerate(zip(cropped_tiles, detections)):
        # Get YOLO detection info
        x1, y1, x2, y2 = map(int, det['bbox'])
        yolo_class = det['class_name']
        yolo_conf = det['confidence']
        
        # Classify using CNN
        try:
            cnn_class, cnn_conf = classifier.classify_tile(tile)
            
            # Draw bounding box with class and confidence
            color = (0, 255, 0)  # Green for high confidence
            if cnn_conf < 0.7:
                color = (0, 165, 255)  # Orange for medium confidence
            if cnn_conf < 0.5:
                color = (0, 0, 255)  # Red for low confidence
                
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"{cnn_class} ({cnn_conf:.2f})"
            if yolo_class != cnn_class:
                label += f" | YOLO: {yolo_class} ({yolo_conf:.2f})"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_image, (x1, y1 - 25), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(output_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = int((x2 - x1) * cnn_conf)
            cv2.rectangle(output_image, (x1, y2), (x1 + bar_width, y2 + 5), color, -1)
            
            # Save cropped tile with classification
            tile_output_path = os.path.join(args.output_dir, f"tile_{i:03d}_{cnn_class}_{cnn_conf:.2f}.jpg")
            cv2.imwrite(tile_output_path, tile)
            
        except Exception as e:
            print(f"Error processing tile {i}: {str(e)}")
    
    # Save and show the result
    output_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.image_path)}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Cropped tiles saved to: {args.output_dir}/")

if __name__ == '__main__':
    main(f'datasets/test/images/000360_jpg.rf.7615ca8e5ba10e9d9639be464cff8f94.jpg')
