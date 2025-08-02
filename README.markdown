# Chest X-Ray Image Classification with Grad-CAM Visualization

## Project Overview
This project implements a deep learning model to classify chest X-ray images (normal vs. pneumonia) with Grad-CAM visualization to highlight the regions of the image that most influenced the model's decision.

## Project Structure
```
XRAY-GRADCAM/
├── app.py                  # Main application file
├── pred_page.py            # Prediction page module
├── cam.py                  # Grad-CAM implementation
├── Grad_cam.py             # Grad-CAM utilities
├── test.py                 # Testing script
├── data/                   # Data directory
│   └── chest_xray/         # Chest X-ray dataset
│       └── raw/            # Raw image data
├── Grad_cam/               # Grad-CAM output directory
│   ├── Figure_1.png        # Sample Grad-CAM visualization 1
│   └── Figure_2.png        # Sample Grad-CAM visualization 2
├── models/                 # Model files
│   ├── model_final.py      # Final model implementation
│   ├── model.py            # Model architecture
│   ├── resnet18_xray.pth   # Pretrained weights
│   └── resnet18_xray2.pth  # Additional pretrained weights
└── myenv/                  # Python virtual environment
```

## Dataset
The project uses the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains:
- 5,863 X-ray images (JPEG format)
- Two classes: Normal and Pneumonia
- Images organized into train, test, and validation sets

## Model Architecture
The project uses a ResNet-18 architecture pretrained on ImageNet and fine-tuned on the chest X-ray dataset. Key features:
- Transfer learning with ResNet-18
- Custom classifier head for binary classification
- Model weights saved in `.pth` files

## Grad-CAM Implementation
The project includes Grad-CAM (Gradient-weighted Class Activation Mapping) visualization to:
- Highlight important regions in the X-ray for diagnosis
- Provide model interpretability
- Generate heatmaps showing where the model focuses when making predictions

## Requirements
To run this project, you'll need:
- Python 3.6+
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- Streamlit (for the web app)

## Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Web App
```bash
streamlit run app.py
```

### Generating Grad-CAM Visualizations
Run the test script:
```bash
python test.py
```

### Training the Model
To train a new model:
```bash
python models/model_final.py
```

## Results
The model achieves:
- High accuracy in classifying normal vs. pneumonia cases
- Interpretable results through Grad-CAM visualizations
- Clear heatmaps showing diagnostically relevant regions

## Future Work
- Expand to multi-class classification (COVID-19, tuberculosis, etc.)
- Implement more advanced visualization techniques
- Develop a more comprehensive web interface for medical professionals

## Acknowledgments
- Dataset provided by Paul Mooney on Kaggle
- ResNet architecture from PyTorch
- Grad-CAM implementation based on the original paper