import sys
import os
import torch
import cv2
import numpy as np
from PIDNet.models.pidnet import get_pred_model
from PIDNet.utils.visualize import decode_segmap
from PIL import Image

# Add the PIDNet directory to the Python path
sys.path.append('/Supplementary/Road_Segmentation/ImageNet1/PIDNet')

def load_model():
    # Load the PIDNet model with the specific pretrained weights
    model = get_pred_model(model_name='pidnet_s', pretrained='C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Supplementary/Road_Segmentation/ImageNet1/weights/PIDNet_S_ImageNet.pth.tar', num_classes=19)
    model.eval()
    return model

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize and normalize the image
    input_image = cv2.resize(image, (1024, 512))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
    return input_image

def segment_image(model, input_image):
    # Perform segmentation
    with torch.no_grad():
        output = model(input_image)[0].argmax(0).cpu().numpy()
    return output

def decode_segmentation(output):
    # Decode the segmentation map
    segmap = decode_segmap(output, dataset='cityscapes')
    return segmap

def display_image(image, segmap):
    # Display the original image and the segmentation map
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Path to the image to be tested
    image_path = '/Supplementary/Road_Segmentation/ImageNet1/test_image.jpg'  # Update this path

    # Load the model
    model = load_model()

    # Preprocess the image
    input_image = preprocess_image(image_path)

    # Perform segmentation
    output = segment_image(model, input_image)

    # Decode the segmentation map
    segmap = decode_segmentation(output)

    # Read the original image for display
    original_image = cv2.imread(image_path)

    # Display the original image and the segmentation map
    display_image(original_image, segmap)

if __name__ == '__main__':
    main()
