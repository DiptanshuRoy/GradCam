import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === Transform function ===
def transform_img():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

# === Load model ===
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load("../models/resnet18_xray2.pth", map_location=torch.device('cpu')))

    model.eval()
    return model

# === Load image ===




image_path = r"../data/chest_xray/archive (1)/chest_xray/train/NORMAL/IM-0135-0001.jpeg"





def image_load():
    image = Image.open(image_path).convert('RGB')
    image = transform_img()(image).unsqueeze(0)
    return image

# === Hooks ===
feature_maps = []
grad = []

def forwardhook(module, input, output):
    feature_maps.clear()
    feature_maps.append(output.detach())

def backwardhook(module, grad_input, grad_output):
    grad.clear()
    grad.append(grad_output[0].detach())

# === Register hooks ===
model = load_model()
target_layer = model.layer4[1].conv2  # last conv layer
target_layer.register_forward_hook(forwardhook)
target_layer.register_backward_hook(backwardhook)

# === Forward pass ===
input_image = image_load()
output = model(input_image)
pred_class = output.argmax(dim=1).item()

# === Backward pass ===
model.zero_grad()
output[0, pred_class].backward()

# === Grad-CAM ===
grads = grad[0]            # [1, C, H, W]
fmap = feature_maps[0]     # [1, C, H, W]
weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
cam = (weights * fmap).sum(dim=1, keepdim=True)  # [1, 1, H, W]
cam = torch.nn.functional.relu(cam)

# === Normalize and resize ===
cam = cam.squeeze().detach().numpy()  # [H, W]
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min())

# === Overlay heatmap ===
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

orig = np.array(Image.open(image_path).resize((224, 224)).convert("RGB")) / 255
overlay = heatmap * 0.4 + orig * 0.6
overlay = np.clip(overlay, 0, 1)

# === Show result ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, cmap='jet')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Overlay on Original")
plt.imshow(overlay)
plt.axis('off')
plt.tight_layout()
plt.show()
