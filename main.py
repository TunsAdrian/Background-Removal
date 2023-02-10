from torchvision import models
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

import os
import torch
import numpy as np
import cv2
import torchvision.transforms as t


def remove_image_background(image, source, nc=21):
    # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # Convert each label to white
    for label in range(1, nc):
        idx = image == label
        r[idx] = g[idx] = b[idx] = [255]

    rgb = np.stack([r, g, b], axis=2)

    foreground = cv2.imread(source)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

    # Match shape of R-band in RGB output map produced by DeepLab V3
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    # Create background array with white pixels
    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return output image for display
    return outImage


def segment(net, image_path):
    image = Image.open(image_path)

    # Images are resized, converted to tensors and normalized with the Imagenet specific mean and standard deviation
    trf = t.Compose([t.Resize(450),
                     t.ToTensor(),
                     t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Convert to [1 x C x H x W] from [C x H x W], because a 'batch' is needed while passing it through the network and move tensor to CUDA
    inp = trf(image).unsqueeze(0).to('cuda')

    # Move model to CUDA and get the result's 'out' key
    out = net.to('cuda')(inp)['out']

    # Obtain 2D images, where each pixel corresponds to a class label (1 - 20), by taking the max index for each pixel position,
    # which represents the class, create a new tensor and move to CPU
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    return remove_image_background(om, image_path)


deeplab = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).eval()

for current_path in os.listdir('./original-images'):
    result = segment(deeplab, './original-images/' + current_path)
    cv2.imwrite('./removed-background-images/' + current_path, cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR))
