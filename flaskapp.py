import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

class VGGModel(nn.Module):
    def __init__(self, num_classes=1):
        super(VGGModel, self).__init__()
        vgg = models.vgg16(pretrained=True)

        # Freeze the pre-trained layers
        for param in vgg.parameters():
            param.requires_grad = False

        self.features = vgg.features

        # Use a dummy input to calculate the output size after features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # VGG expects 224x224 input size
            dummy_output = self.features(dummy_input)
            num_features = dummy_output.numel() // dummy_input.size(0)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return torch.sigmoid(x)

# Define data transformations and loaders
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Load the PyTorch model (e.g., ResNet18)
@st.cache(allow_output_mutation=True)  # Cache the model loading to avoid reloading every time
def load_model():
    model = VGGModel()  # Using resnet18 for this example
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Function to make predictions
def predict(image, model):
    # Apply the transformations to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

# Streamlit app title and description
st.title("Brain Tumor Image Classifier")
st.write("Upload an image and let the model classify it.")

# File uploader allows user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Load the model
    model = load_model()
    
    # Run the prediction function
    input_image = preprocess_transform(image)
    input_image = input_image.unsqueeze(0)  # Add batch dimension

# Perform inference
    with torch.no_grad():
        output = model(input_image)

# Interpret the result
    prediction = torch.round(output).item()

    if prediction == 1:
        class_label = 'yes (tumor detected)'
    else:
        class_label = 'no (no tumor detected)'

    st.write(f"Prediction: {class_label}")
    
    # Display the predicted class (for now, just showing the predicted class index)
    #st.write(f"Predicted Class: {name_disease}")
