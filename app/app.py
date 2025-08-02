MODEL_PATH = "models/resnet18_xray2.pth"  # UPDATE THIS
IMAGE_PATH = "data/archive (1)/chest_xray/train/PNEUMONIA/person1_bacteria_2.jpg"  # UPDATE THIS

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 2)
)

# Try to load trained weights, otherwise use random
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Loaded model from {MODEL_PATH}")
except:
    print("⚠ Using random weights - results won't be meaningful")

model.to(device).eval()

# Load and preprocess image
image = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])
input_tensor = preprocess(image).unsqueeze(0).to(device)

# Grad-CAM setup
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.clear()
    feature_maps.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0].detach())

# Register hooks
model.layer4.register_forward_hook(forward_hook)
model.layer4.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
print(f"Predicted class: {pred_class}")

# Backward pass
model.zero_grad()
output[0, pred_class].backward()

# Compute Grad-CAM
weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
cam = torch.sum(weights * feature_maps[0], dim=1)
cam = torch.relu(cam)
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = cam.squeeze().cpu().numpy()

# Visualize
cam_resized = cv2.resize(cam, image.size)
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(heatmap, 0.5, np.array(image), 0.5, 0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='hot')
plt.title('Grad-CAM')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title('Overlay')
plt.axis('off')

plt.show()
