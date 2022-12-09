import cv2

def rgb_to_hsv(im: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

def extract_channel(hsv: cv2.Mat, channel='hue'):

    h, s, v = cv2.split(hsv)

    map: dict[str, cv2.Mat] = {
        'hue': h,
        'sat': s,
        'val': v
    }

    return map[channel]

def extract_hue(im: cv2.Mat, is_hsv=False):
    hsv: cv2.Mat = im.copy()
    if not is_hsv:
        hsv: cv2.Mat = rgb_to_hsv(im)
    return extract_channel(hsv, 'hue')

def extract_sat(im: cv2.Mat, is_hsv=False):
    hsv: cv2.Mat = im.copy()
    if not is_hsv:
        hsv: cv2.Mat = rgb_to_hsv(im)
    return extract_channel(hsv, 'sat')

def extract_val(im: cv2.Mat, is_hsv=False):
    hsv: cv2.Mat = im.copy()
    if not is_hsv:
        hsv: cv2.Mat = rgb_to_hsv(im)
    return extract_channel(hsv, 'val')