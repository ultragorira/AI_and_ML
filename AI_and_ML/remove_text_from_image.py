#!/usr/bin/env python3

"""
Remove text from an image
"""
from argparse import ArgumentParser, Namespace
import keras_ocr
import cv2
import math
import numpy as np
from typing import Tuple, List

def read_image(image_path: str) -> Tuple[List, np.ndarray, np.ndarray]:
    """Read Image in Keras OCR and recognize texts"""
    pipeline = keras_ocr.pipeline.Pipeline()
    img = keras_ocr.tools.read(image_path) 
    predictions = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")

    return predictions, mask, img

def inpaint(predictions: List, mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """Function to inpaint text in an image"""

    for box in predictions[0]:
        x1, y1 = box[1][0]
        x2, y2 = box[1][1] 
        x3, y3 = box[1][2]
        x4, y4 = box[1][3] 
        
        x_mid1, y_mid1 = midpoint(x2 , y2, x3, y3)
        x_mid2, y_mid2 = midpoint(x1, y1, x4, y4)
        
        line_thickness = int(math.sqrt( (x3 - x2)**2 + (y3 - y2)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid1 - int((x_mid1 * 0.02)), y_mid1), (x_mid2 + int((x_mid2 * 0.02)), y_mid2), 255, line_thickness)
        inpainted_img = cv2.inpaint(original_image, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)


def midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    """Calculate mid points"""
    x_mid = int(np.divide((x1 + x2), 2))
    y_mid = int(np.divide((y1 + y2), 2))
    return (x_mid, y_mid)

def main() -> None:

    args = parse_args()
    predictions, mask, original_image = read_image(args.image_path)
    processed_image = inpaint(predictions, mask, original_image)
    cv2.imwrite('processed.jpg', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    print("Done!")

def parse_args() -> Namespace:

    parser = ArgumentParser(prog = "Remove text from image", description = __doc__)
    parser.add_argument("image_path", 
                        help = 'Pah to image')

    return parser.parse_args()

if __name__ == "__main__":
    main()