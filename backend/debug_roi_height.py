import cv2
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omr_engine import OMREngine

def debug_roi_height():
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (1).tif"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    engine = OMREngine()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract ROI logic
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
        h, w = warped.shape
        print(f"ROI Height: {h}, Width: {w}")
        
        # Visualize Grid
        debug_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        
        # Vertical Grid (15 rows)
        row_height = h / 15.0
        print(f"Calculated Row Height: {row_height}")
        
        for i in range(15):
            y = int(i * row_height)
            cv2.line(debug_img, (0, y), (w, y), (0, 255, 0), 1)
            cv2.putText(debug_img, f"R{i+1}", (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
        cv2.imwrite("debug_roi_grid.jpg", debug_img)
        print("Saved debug_roi_grid.jpg")

if __name__ == "__main__":
    debug_roi_height()
