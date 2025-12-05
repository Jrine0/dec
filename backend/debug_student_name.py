import cv2
import numpy as np
from omr_engine import OMREngine

def debug_name():
    engine = OMREngine()
    img_path = r"c:\Users\jitin\Desktop\Desk\drive\Documents\omr\OMR_SRS\image (1).tif"
    img = cv2.imread(img_path)
    
    # Extract fields to get the Name ROI
    # We'll hack into extract_fields or just copy the logic
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge Detection to find blocks
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Name is usually the 2nd largest
    name_contour = cnts[1]
    x, y, w, h = cv2.boundingRect(name_contour)
    print(f"Name ROI: {x},{y},{w},{h}")
    
    roi = gray[y:y+h, x:x+w]
    cv2.imwrite("debug_name_roi_raw.jpg", roi)
    
    # Process Text Block Logic
    # Standard threshold, NO morphology
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    cv2.imwrite("debug_name_roi_thresh_nomorph.jpg", thresh)
    
    # ROI-based Grid Sampling
    # We assume the ROI covers the 20 columns and 26 rows (A-Z)
    # Dimensions: w=1154, h=1538
    
    num_cols = 20
    num_rows = 28 # Assume 2 header rows
    
    col_w = w / num_cols
    row_h = h / num_rows
    
    print(f"Grid: {num_cols}x{num_rows}, Cell: {col_w:.1f}x{row_h:.1f}")
    
    result = ""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Use the thresholded image for sampling
    # Revert to standard threshold
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    for c_idx in range(num_cols):
        # Define Column X range
        x1 = int(c_idx * col_w)
        x2 = int((c_idx + 1) * col_w)
        
        best_r_idx = -1
        max_ratio = 0.0
        
        for r_idx in range(num_rows):
            # Define Row Y range
            y1 = int(r_idx * row_h)
            y2 = int((r_idx + 1) * row_h)
            
            # Crop cell
            # Add a small margin/padding to avoid grid lines? 
            # Or shrink the crop to sample the center?
            # Let's shrink by 20%
            pad_x = int(col_w * 0.2)
            pad_y = int(row_h * 0.2)
            
            cell = thresh[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
            
            if cell.size == 0: continue
            
            nz = cv2.countNonZero(cell)
            area = cell.size
            ratio = nz / area
            
            if ratio > max_ratio:
                max_ratio = ratio
                best_r_idx = r_idx
                
        print(f"Col {c_idx}: Max Ratio {max_ratio:.2f} at Row {best_r_idx} ({alphabet[best_r_idx] if best_r_idx >=0 else '-'})")
        
        if max_ratio > 0.4: # Threshold
            if best_r_idx != -1:
                result += alphabet[best_r_idx]
    # Draw Grid
    debug_grid_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    for c_idx in range(num_cols + 1):
        x = int(c_idx * col_w)
        cv2.line(debug_grid_img, (x, 0), (x, int(h)), (0, 255, 0), 1)
        
    for r_idx in range(num_rows + 1):
        y = int(r_idx * row_h)
        cv2.line(debug_grid_img, (0, y), (int(w), y), (0, 0, 255), 1)
        
    cv2.imwrite("debug_name_grid.jpg", debug_grid_img)
    
    print(f"Result: '{result}'")
    print(f"Collapsed: '{' '.join(result.split())}'")

if __name__ == "__main__":
    debug_name()
