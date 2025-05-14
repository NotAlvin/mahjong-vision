import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

class MahjongTileDetector:
    """Class for detecting mahjong tiles using YOLO."""
    
    def __init__(self, model_path=f'runs/detect/mahjong_detector7/weights/best.pt'):
        """
        Initialize the YOLO model for mahjong tile detection.
        
        Args:
            model_path: Path to a pre-trained YOLO model. If None, uses YOLOv8n.
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Load a pre-trained YOLOv8n model
            self.model = YOLO('yolov8n.pt')
            print("Using pre-trained YOLOv8n model. For best results, fine-tune on mahjong tile data.")
    
    def detect_tiles(self, image_path, conf_threshold=0.25):
        """
        Detect mahjong tiles in an image.
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold for detection
            
        Returns:
            tuple: (cropped_tiles, detections, img) where:
                - cropped_tiles: List of cropped tile images
                - detections: List of dictionaries containing detection info (bbox, confidence, class_id, class_name)
                - img: The original image
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Run YOLO detection
        results = self.model(img, conf=conf_threshold)
        
        # Extract detections
        boxes = results[0].boxes
        
        # Process detections
        cropped_tiles = []
        detections = []
        
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
            
            # Crop the tile
            crop = img[y1:y2, x1:x2]
            
            # Store results
            cropped_tiles.append(crop)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
        
        return cropped_tiles, detections, img
    
    def visualize_detections(self, image_path, conf_threshold=0.25):
        """
        Visualize the detected tiles on the image.
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold for detection
        """
        # Run detection
        _, bboxes, img = self.detect_tiles(image_path, conf_threshold)
        
        # Draw bounding boxes
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Tile {i+1}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Detected Mahjong Tiles')
        plt.show()
        
        return img

# Example usage

def example_usage():
    """Example of how to use the mahjong recognition system."""
    # Create the system
    system = MahjongTileDetector()
    
    # Process an image
    image_path = 'datasets/test/images/000063_jpg.rf.e6ec1b5ff061f4c6ee4d9018ff67b8ef.jpg'
    cropped_tiles, detections, img = system.detect_tiles(image_path)
    
    # Print results and save cropped tiles
    print(f"Found {len(cropped_tiles)} mahjong tiles:")
    for i, (tile, detection) in enumerate(zip(cropped_tiles, detections)):
        print(f"  Tile {i+1}: {detection['class_name']} (Confidence: {detection['confidence']:.2f})")
        
        # Save the cropped tile with label
        label = f"{detection['class_name']}_{i+1}"
        cv2.imwrite(f'mahjong_tile_{i+1}_{label}.jpg', tile)
    
    # Save the annotated image
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('mahjong_detected.jpg', img)
    print("Saved annotated image as 'mahjong_detected.jpg'")


# Script to fine-tune the YOLO model on mahjong tile data
def fine_tune_detector():
    """
    Fine-tune YOLO model on mahjong tile detection data.
    """
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model with the mahjong dataset
    results = model.train(
        data=f'data.yaml',
        epochs=32,
        imgsz=640,
        batch=16,
        name='mahjong_detector',
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save the model
    model_path = f'mahjong_detector/weights/best.pt'
    print(f"Model saved at {model_path}")
    return model_path


if __name__ == "__main__":
    # This would be your entry point for running the application
    #fine_tune_detector()
    example_usage()