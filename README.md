# Handwritten Text Segmentation Project

This project focuses on segmenting handwritten text from images using deep learning. The goal is to create a robust segmentation model trained on synthetic data and evaluate its performance on real-world handwriting samples.

## Project Overview

The project is divided into the following phases:

1. **Synthetic Dataset Creation**:
   - Generate synthetic handwritten text using diverse fonts and text sources.
   - Render text onto blank images with randomized placements and styles.
   - Apply data augmentations (e.g., rotation, noise, blurring) to simulate real-world variations.

2. **Segmentation Mask Generation**:
   - Create approximate segmentation masks using text placement coordinates.
   - Refine masks using template matching for precise character localization.

3. **Model Training**:
   - Train a segmentation model (e.g., U-Net, SwinTransformer) on the synthetic dataset.
   - Use metrics like Intersection over Union (IoU) and Dice Score to evaluate performance.

4. **Evaluation on Real Data**:
   - Test the trained model on real-world handwriting samples.
   - Analyze results and identify areas for improvement.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-text-segmentation.git
   cd handwritten-text-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Synthetic Dataset**:
   - Run `scripts/preprocess_text.py` to clean and prepare text data.
   - Use `scripts/render_text.py` to render text onto images.
   - Apply augmentations with `scripts/augment_images.py`.

2. **Create Segmentation Masks**:
   - Generate approximate masks using text placement coordinates.
   - Refine masks using template matching.

3. **Train the Model**:
   - Use the scripts in `models/` to train a segmentation network (e.g., U-Net).

4. **Evaluate on Real Data**:
   - Preprocess real handwriting samples and run predictions using the trained model.

## Tools and Technologies

- **Libraries**: PIL, OpenCV, ImgAug, PyTorch.
- **Models**: U-Net.
- **Annotation Tools**: LabelIng (for manual mask validation).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](License) for details.
