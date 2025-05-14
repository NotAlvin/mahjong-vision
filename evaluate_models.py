import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import seaborn as sns
from yolo_detector import MahjongTileDetector
from tile_classifier import MahjongTileClassifier

class ModelEvaluator:
    def __init__(self, detector_weights=f'runs/detect/mahjong_detector7/weights/best.pt', classifier_weights=f'best_model.pth'):
        """
        Initialize the model evaluator with trained models.
        
        Args:
            detector_weights: Path to trained YOLO detector weights
            classifier_weights: Path to trained classifier weights
        """
        # Initialize detector
        self.detector = MahjongTileDetector(model_path=detector_weights) if detector_weights else MahjongTileDetector()
        
        # Initialize classifier
        self.classifier = MahjongTileClassifier(model_path=classifier_weights) if classifier_weights else MahjongTileClassifier()
        print(self.classifier)
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.model = self.classifier.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def load_ground_truth(self, label_path):
        """Load ground truth labels from a YOLO format label file."""
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            return []
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        ground_truths = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_id = int(parts[0])
            # Convert YOLO format (normalized points) to pixel coordinates
            points = [float(x) for x in parts[1:]]
            
            # Ensure we have valid points (should be 8 values: x1,y1,x2,y2,x3,y3,x4,y4)
            if len(points) != 8:
                print(f"Warning: Invalid number of points in {label_path}: {line.strip()}")
                continue
                
            ground_truths.append({
                'class_id': class_id,
                'points': points
            })
            
        return ground_truths
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert points to (x1, y1, x2, y2) format
        def get_coords(points):
            if len(points) == 4:  # Already in [x1,y1,x2,y2] format
                return points[0], points[1], points[2], points[3]
            # For polygon points [x1,y1,x2,y2,x3,y3,x4,y4]
            x_coords = points[0::2]
            y_coords = points[1::2]
            return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        
        try:
            box1_coords = get_coords(box1)
            box2_coords = get_coords(box2)
            
            # Calculate intersection
            x_left = max(box1_coords[0], box2_coords[0])
            y_top = max(box1_coords[1], box2_coords[1])
            x_right = min(box1_coords[2], box2_coords[2])
            y_bottom = min(box1_coords[3], box2_coords[3])
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas
            box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
            box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
            union_area = box1_area + box2_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0.0
            return max(0.0, min(1.0, iou))  # Clamp between 0 and 1
            
        except Exception as e:
            print(f"Error in calculate_iou: {e}")
            print(f"box1: {box1}")
            print(f"box2: {box2}")
            return 0.0
    
    def evaluate_detector(self, test_dir, iou_threshold=0.5, debug=False):
        """
        Evaluate YOLO detector performance.
        
        Args:
            test_dir: Directory containing 'images' and 'labels' subdirectories
            iou_threshold: IoU threshold to consider a detection as correct
            debug: If True, print debug information for first few images
            
        Returns:
            Dictionary with detection metrics
        """
        test_dir = Path(test_dir)
        image_files = list((test_dir / 'images').glob('*.jpg'))
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_ground_truth = 0
        debug_count = 3  # Number of images to debug
        
        # For class name to ID mapping
        class_id_map = {v: k for k, v in self.classifier.TILE_CLASSES.items()}
        
        for img_path in tqdm(image_files, desc="Evaluating Detector"):
            # Load image to get dimensions
            image = cv2.imread(str(img_path))
            img_height, img_width = image.shape[:2]
            
            # Load ground truth
            label_path = test_dir / 'labels' / f"{img_path.stem}.txt"
            ground_truths = self.load_ground_truth(label_path)
            total_ground_truth += len(ground_truths)
            
            if debug and debug_count > 0:
                print(f"\n=== Debugging {img_path.name} ===")
                print(f"Image size: {img_width}x{img_height}")
                print(f"Ground truths: {len(ground_truths)}")
                for i, gt in enumerate(ground_truths):
                    points = np.array(gt['points']).reshape(-1, 2) * [img_width, img_height]
                    print(f"  GT {i}: class={gt['class_id']}, points={points.astype(int).flatten().tolist()}")
            
            # Run detection with debug info
            try:
                _, detections, _ = self.detector.detect_tiles(str(img_path), conf_threshold=0.25)
                if debug and debug_count > 0:
                    print(f"Detections: {len(detections)}")
                    for i, det in enumerate(detections):
                        bbox = det['bbox']
                        print(f"  Det {i}: class={det.get('class_name', 'N/A')} (ID: {det.get('class_id', 'N/A')}), "
                              f"conf={det.get('confidence', 0):.2f}, bbox={[int(x) for x in bbox]}")
            except Exception as e:
                print(f"Error during detection on {img_path}: {e}")
                detections = []
            
            if debug and debug_count > 0:
                print(f"Detections: {len(detections)}")
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    print(f"  Det {i}: class={det['class_name']}, conf={det['confidence']:.2f}, bbox={[int(x) for x in bbox]}")
            
            # Match detections with ground truth
            matched_gt = set()
            
            for det in detections:
                # Get detection box in [x1,y1,x2,y2] format
                det_box = det['bbox']
                det_points = [det_box[0], det_box[1], det_box[2], det_box[3]]
                
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(ground_truths):
                    if i in matched_gt:
                        continue
                        
                    # Get ground truth points (already in [x1,y1,x2,y2,x3,y3,x4,y4] format)
                    gt_points = gt['points']
                    
                    # Convert to pixel coordinates if needed
                    if max(gt_points) <= 1.0:  # If normalized
                        gt_points = np.array(gt_points).reshape(-1, 2) * [img_width, img_height]
                        gt_points = gt_points.flatten().tolist()
                    
                    iou = self.calculate_iou(det_points, gt_points)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    # Get the class ID from the detection (YOLO might return string or int)
                    det_class_id = det.get('class_id')
                    if det_class_id is None and 'class_name' in det:
                        # If we have class name but not ID, try to map it
                        det_class_id = class_id_map.get(det['class_name'], -1)
                    
                    gt_class_id = ground_truths[best_gt_idx]['class_id']
                    
                    # Compare class IDs
                    if int(det_class_id) == int(gt_class_id):
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        if debug and debug_count > 0:
                            print(f"  Match: Det {i} -> GT {best_gt_idx}, Class ID: {gt_class_id}, IoU={best_iou:.2f}")
                    else:
                        false_positives += 1
                        if debug and debug_count > 0:
                            print(f"  Class Mismatch: Det class ID {det_class_id} != GT class ID {gt_class_id}, IoU={best_iou:.2f}")
                else:
                    false_positives += 1
                    if debug and debug_count > 0:
                        print(f"  Low IoU: Det {i}, max IoU={best_iou:.2f} < {iou_threshold}")
                        
            if debug and debug_count > 0:
                debug_count -= 1
            
            # Count unmatched ground truths as false negatives
            unmatched_gt = len(ground_truths) - len(matched_gt)
            false_negatives += unmatched_gt
            if debug and debug_count > 0:
                if unmatched_gt > 0:
                    print(f"  False Negatives: {unmatched_gt} ground truths not detected")
                if len(matched_gt) > 0:
                    print(f"  True Positives: {len(matched_gt)} matched detections")
                if len(detections) > len(matched_gt):
                    print(f"  False Positives: {len(detections) - len(matched_gt)} unmatched detections")
                print(f"  Current metrics: TP={true_positives}, FP={false_positives}, FN={false_negatives}")
            
            debug_count -= 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_classifier(self, test_dir, use_yolo_classifier=False, min_confidence=0.5):
        """Evaluate the classifier on test images.
        
        Args:
            test_dir: Directory containing 'images' and 'labels' subdirectories
            use_yolo_classifier: If True, use YOLO's built-in classifier. 
                              If False, use the CNN classifier.
            min_confidence: Minimum confidence threshold for predictions
                                  
        Returns:
            Dictionary with classification metrics
        """
        test_dir = Path(test_dir)
        # Get all test images and limit to first 5 for debugging
        test_images = sorted(list((test_dir / 'images').glob('*.jpg')) + list((test_dir / 'images').glob('*.png')))[:5]
        if not test_images:
            print(f"No test images found in {test_dir}")
            return {'accuracy': 0.0, 'report': {}, 'confusion_matrix': None, 'y_true': [], 'y_pred': []}
        
        print(f"Processing first {len(test_images)} images for debugging...")
        
        y_true = []
        y_pred = []
        
        for img_path in tqdm(test_images, desc="Evaluating Classifier"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Load ground truth
            label_path = test_dir / 'labels' / f"{img_path.stem}.txt"
            ground_truths = self.load_ground_truth(label_path)
            if not ground_truths:
                continue
            
            # Run detection if using YOLO classifier or for matching
            try:
                _, detections, _ = self.detector.detect_tiles(str(img_path), conf_threshold=0.25)
            except Exception as e:
                print(f"Error detecting tiles in {img_path}: {e}")
                continue
        
            # For each ground truth, classify the corresponding region
            for gt in ground_truths:
                class_id = gt['class_id']
                points = np.array(gt['points']).reshape(-1, 2) * [image.shape[1], image.shape[0]]
                
                # Get bounding box from points
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                
                # Add some padding (10% of width/height, but stay within image bounds)
                h, w = image.shape[:2]
                pad_x = int(0.1 * (x2 - x1))
                pad_y = int(0.1 * (y2 - y1))
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                # Extract and check crop
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < 10:
                    print(f"Skipping small or empty crop in {img_path}")
                    continue
                    
                # Debug: Save the crop with ground truth label
                debug_dir = Path('debug_crops')
                debug_dir.mkdir(exist_ok=True)
                
                # Create a copy of the crop for visualization
                crop_debug = crop.copy()
                
                # Add ground truth label to the image
                gt_label = self.classifier.TILE_CLASSES.get(class_id, str(class_id))
                cv2.putText(crop_debug, f"GT: {gt_label}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save the debug image
                debug_path = debug_dir / f"{img_path.stem}_crop_{len(y_true)}.jpg"
                cv2.imwrite(str(debug_path), cv2.cvtColor(crop_debug, cv2.COLOR_RGB2BGR))
                
                try:
                    if use_yolo_classifier:
                        # For YOLO, we still need to find the best matching detection
                        # but we'll use the ground truth crop for classification
                        best_iou = 0
                        best_det = None
                        
                        for det in detections:
                            det_box = det['bbox']
                            det_points = [
                                det_box[0], det_box[1], 
                                det_box[2], det_box[1], 
                                det_box[2], det_box[3], 
                                det_box[0], det_box[3]
                            ]
                            
                            iou = self.calculate_iou(points.flatten().tolist(), det_points)
                            if iou > best_iou and iou > 0.5:  # Require minimum IoU of 0.5
                                best_iou = iou
                                best_det = det
                        
                        if best_det is not None and best_det.get('confidence', 0) >= min_confidence:
                            # Use the ground truth crop for classification
                            pred_class, confidence = self.classifier.classify_tile(crop)
                            if confidence >= min_confidence and pred_class != -1:
                                y_true.append(class_id)
                                y_pred.append(pred_class)
                    else:
                        # Use CNN classifier with the same ground truth crop
                        pred_class, confidence = self.classifier.classify_tile(crop)
                        pred_label = self.classifier.TILE_CLASSES.get(pred_class, str(pred_class))
                        print(f'Predicted: {pred_label}, Actual: {gt_label}, Confidence: {confidence}')
                        if confidence >= min_confidence and pred_class != -1:
                            # Add prediction to the debug image
                            cv2.putText(crop_debug, f"Pred: {pred_label} ({confidence:.2f})", (5, 40), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.imwrite(str(debug_path), cv2.cvtColor(crop_debug, cv2.COLOR_RGB2BGR))
                            
                            y_true.append(class_id)
                            y_pred.append(pred_class)
                except Exception as e:
                    print(f"Error processing tile in {img_path}: {e}")
                    continue
        # Calculate metrics
        if not y_true:  # No valid predictions
            return {
                'accuracy': 0.0,
                'report': {},
                'confusion_matrix': None,
                'y_true': [],
                'y_pred': []
            }
        
        # Get the list of actual classes that appear in the data
        unique_classes = sorted(TILE_CLASSES.keys())
        class_names = [self.classifier.TILE_CLASSES.get(c, str(c)) for c in unique_classes]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            labels=unique_classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        
        print(f"\nTotal predictions: {len(y_pred)}")
        print(f"Unique classes predicted: {len(set(y_pred))}")
        print(f"Unique true classes: {len(set(y_true))}")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }

    def plot_confusion_matrix(self, cm, class_names, title='Confusion Matrix'):
        """Plot confusion matrix with the given title."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def main():    
    parser = argparse.ArgumentParser(description='Evaluate Mahjong Tile Detection and Classification Models')
    parser.add_argument('--test_dir', type=str, default='datasets/test',
                       help='Path to test dataset directory')
    parser.add_argument('--detector_weights', type=str, default=None,
                       help='Path to trained YOLO detector weights')
    parser.add_argument('--classifier_weights', type=str, default=None,
                       help='Path to trained classifier weights')
    args = parser.parse_args()
    
    # Initialize evaluator
    print("Initializing models...")
    evaluator = ModelEvaluator(
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights
    )
    
    # Evaluate detector with debug info
    # print("\n=== Evaluating Detector ===")
    # detector_metrics = evaluator.evaluate_detector(args.test_dir, debug=True)
    
    # print("\nDetection Metrics:")
    # print(f"True Positives: {detector_metrics['true_positives']}")
    # print(f"False Positives: {detector_metrics['false_positives']}")
    # print(f"False Negatives: {detector_metrics['false_negatives']}")
    # print(f"Precision: {detector_metrics['precision']:.4f}")
    # print(f"Recall: {detector_metrics['recall']:.4f}")
    # print(f"F1 Score: {detector_metrics['f1_score']:.4f}")
    
    # Evaluate CNN classifier
    print("\n=== Evaluating CNN Classifier ===")
    cnn_metrics = evaluator.evaluate_classifier(args.test_dir, use_yolo_classifier=False)
    
    if not cnn_metrics['y_true']:
        print("\nNo valid predictions were made during evaluation.")
        print("This could be due to no tiles being detected or all predictions being below the confidence threshold.")
    else:
        print(f"\nCNN Classification Accuracy: {cnn_metrics['accuracy']:.4f}")
        print("\nCNN Classification Report:")
        
        # Get unique classes that appear in either y_true or y_pred
        unique_classes = sorted(set(cnn_metrics['y_true'] + cnn_metrics['y_pred']))
        class_names = [evaluator.classifier.TILE_CLASSES.get(c, str(c)) for c in unique_classes]
        
        print(classification_report(
            cnn_metrics['y_true'], 
            cnn_metrics['y_pred'],
            labels=unique_classes,
            target_names=class_names,
            zero_division=0
        ))
    
    # Plot CNN confusion matrix
    evaluator.plot_confusion_matrix(
        cnn_metrics['confusion_matrix'],
        list(evaluator.classifier.TILE_CLASSES.values()),
        'CNN Classifier Confusion Matrix'
    )
    
    # Evaluate YOLO classifier
    print("\n=== Evaluating YOLO Classifier ===")
    yolo_metrics = evaluator.evaluate_classifier(args.test_dir, use_yolo_classifier=True)
    
    print(f"\nYOLO Classification Accuracy: {yolo_metrics['accuracy']:.4f}")
    print("\nYOLO Classification Report:")
    print(classification_report(
        yolo_metrics['y_true'],
        yolo_metrics['y_pred'],
        target_names=list(evaluator.classifier.TILE_CLASSES.values()),
        zero_division=0
    ))
    
    # Plot YOLO confusion matrix
    evaluator.plot_confusion_matrix(
        yolo_metrics['confusion_matrix'],
        list(evaluator.classifier.TILE_CLASSES.values()),
        'YOLO Classifier Confusion Matrix'
    )
    
    # Compare results
    print("\n=== Model Comparison ===")
    print(f"{'Metric':<20} {'CNN':<10} {'YOLO':<10}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {cnn_metrics['accuracy']:.4f}    {yolo_metrics['accuracy']:.4f}")
    print(f"{'Precision (weighted)':<20} {cnn_metrics['report']['weighted avg']['precision']:.4f}    {yolo_metrics['report']['weighted avg']['precision']:.4f}")
    print(f"{'Recall (weighted)':<20} {cnn_metrics['report']['weighted avg']['recall']:.4f}    {yolo_metrics['report']['weighted avg']['recall']:.4f}")
    print(f"{'F1-Score (weighted)':<20} {cnn_metrics['report']['weighted avg']['f1-score']:.4f}    {yolo_metrics['report']['weighted avg']['f1-score']:.4f}")
    
    # Save results to file
    with open('evaluation_results.txt', 'w') as f:
        # f.write("=== Detection Metrics ===\n")
        # f.write(f"True Positives: {detector_metrics['true_positives']}\n")
        # f.write(f"False Positives: {detector_metrics['false_positives']}\n")
        # f.write(f"False Negatives: {detector_metrics['false_negatives']}\n")
        # f.write(f"Precision: {detector_metrics['precision']:.4f}\n")
        # f.write(f"Recall: {detector_metrics['recall']:.4f}\n")
        # f.write(f"F1 Score: {detector_metrics['f1_score']:.4f}\n\n")
        
        f.write("=== Classification Report ===\n")
        f.write(classification_report(
            cnn_metrics['y_true'],
            cnn_metrics['y_pred'],
            target_names=list(evaluator.classifier.TILE_CLASSES.values()),
            zero_division=0
        ))
        
        f.write("=== YOLO Classification Report ===\n")
        f.write(classification_report(
            yolo_metrics['y_true'],
            yolo_metrics['y_pred'],
            target_names=list(evaluator.classifier.TILE_CLASSES.values()),
            zero_division=0
        ))
    
    print("\nEvaluation complete! Results saved to 'evaluation_results.txt' and 'confusion_matrix.png'")

if __name__ == '__main__':
    main()
