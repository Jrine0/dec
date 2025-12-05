import cv2
import numpy as np
from omr_engine import OMREngine
import os

def apply_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def debug_contrast(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    srs_dir = os.path.join(os.path.dirname(base_dir), "OMR_SRS")
    path = os.path.join(srs_dir, filename)
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Processing {filename}...")
    engine = OMREngine()
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract Answer ROI (simplified from engine)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    answer_contour = cnts[0]
    peri = cv2.arcLength(answer_contour, True)
    approx = cv2.approxPolyDP(answer_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
         warped = engine.four_point_transform(gray, approx.reshape(4, 2))
         
         # 1. Original
         analyze_roi(warped, "Original")
         
         # 2. Gamma Correction (Darken midtones)
         gamma_img = apply_gamma(warped, gamma=0.5)
         analyze_roi(gamma_img, "Gamma 0.5")
         
         # 3. CLAHE
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
         clahe_img = clahe.apply(warped)
         analyze_roi(clahe_img, "CLAHE")
         
         # 4. Gamma + CLAHE
         combo = clahe.apply(gamma_img)
         analyze_roi(combo, "Gamma+CLAHE")

def analyze_roi(img, label):
    # Thresholding
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Sample a known empty area vs filled area (approx coords from previous debug)
    # Q1 (Filled D): ~1924, 110
    # Q2 (Empty D): ~1924, 165 (approx)
    
    # Let's just calculate global fill percentage as a proxy for noise
    nz = cv2.countNonZero(thresh)
    total = thresh.size
    ratio = nz / total
    print(f"[{label}] Global Fill Ratio: {ratio:.4f}")
    
    cv2.imwrite(f"debug_contrast_{label}.jpg", thresh)

if __name__ == "__main__":
    debug_contrast("image (1).tif")
