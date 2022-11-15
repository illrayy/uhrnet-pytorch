from uhrnet import UHRnet_Segmentation
from PIL import Image
import cv2

img_path  = 'img/street.jpg'

if __name__ == "__main__":
    hrnet = UHRnet_Segmentation()

    image       = Image.open(img_path)
    img_predict = hrnet.detect_image(image)

    cv2.imshow('mask', img_predict)
    cv2.waitKey(0)

    