import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# define the CNN model 
def create_model(num_classes):
    model = models.efficientnet_b2(weights = "IMAGENET1K_V1")
    #freeze the layers
    for name, param in model.named_parameters():
        if 'features.0' in name or 'features.1' in name:
            param.requires_grad = False
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )

    return model

# Load the pre-trained model
checkpoint = torch.load('CNN_model/waste_classification_model.pth', map_location=torch.device('cpu'))
waste_classes = checkpoint['class_names']
num_classes = len(waste_classes)

model = create_model(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy waste classes
# waste_classes = ['Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted = torch.argmax(outputs, 1).item()
        predicted_class = waste_classes[predicted % len(waste_classes)]
    return predicted_class, "Recyclable" if predicted_class in ['Plastic', 'Metal', 'Glass', 'Paper'] else "Non-recyclable"
