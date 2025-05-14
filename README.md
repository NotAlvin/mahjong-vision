# Mahjong Vision

Mahjong Vision is a project designed to create a pipeline for detecting and classifying mahjong tiles in images. The primary goal is to accurately draw boundaries around mahjong tiles and classify their types. This technology can be used as part of a scoring system for various mahjong games.

## Features

- **Tile Detection**: Utilizes a YOLO-based detector to identify and crop mahjong tiles from images.
- **Tile Classification**: Employs a CNN model to classify the type of mahjong tile from the cropped images.
- **Test-Time Augmentation**: Enhances classification accuracy through multiple transformations during inference.
- **Training Pipeline**: Provides a comprehensive training routine with data augmentation and validation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NotAlvin/mahjong-vision.git
   cd mahjong-vision
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the Mahjong Tile Classifier, run:
```bash
python tile_classifier.py --data_root datasets --epochs 50 --batch_size 64
```

### Testing

To test the model on a sample image:
```bash
python tile_classifier.py --test_image path/to/image.jpg
```

## Future Applications

- Integration into mahjong game scoring systems.
- Real-time tile recognition in video streams.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch for providing the deep learning framework.
- OpenCV and PIL for image processing capabilities.
- YOLO for the detection model framework.
