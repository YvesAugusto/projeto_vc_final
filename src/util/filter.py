import cv2
import numpy as np
from .image import extract_hue, extract_sat, extract_val, rgb_to_hsv

def hsv_generic_filter(
    hsv: cv2.Mat,
    lower: list[int],
    upper: list[int]
):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask

def hsv_filter_hue(
    hsv: cv2.Mat,
    lower: int,
    upper: int
):
    return hsv_generic_filter(hsv, [lower, 0, 0], [upper, 255, 255])

def hsv_filter_sat(
    hsv: cv2.Mat,
    lower: int,
    upper: int
):
    return hsv_generic_filter(hsv, [0, lower, 0], [179, upper, 255])

def hsv_filter_val(
    hsv: cv2.Mat,
    lower: int,
    upper: int
):
    return hsv_generic_filter(hsv, [0, 0, lower], [179, 255, upper])