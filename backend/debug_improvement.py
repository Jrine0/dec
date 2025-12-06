import cv2
import numpy as np
from omr_engine import OMREngine
import os

def debug_image(filename):
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
    
    # We need to access internal methods or modify engine to return debug info
    # For now, let's copy the relevant parts of extract_fields here to debug
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    if len(cnts) < 5:
        print("Not enough contours.")
        return

    # Identify ROIs
    answer_contour = cnts[0]
    name_contour = cnts[1]
    
    top_blocks = []
    for c in cnts[2:]:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if 100000 < area < 200000 and y < 1000:
            top_blocks.append(c)
            
    top_blocks, _ = engine.sort_contours(top_blocks, method="left-to-right")
    
    print(f"Top Blocks Found: {len(top_blocks)}")
    for i, c in enumerate(top_blocks):
        print(f"Block {i}: Area={cv2.contourArea(c)}, Rect={cv2.boundingRect(c)}")
        
        # Save ROI for inspection
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y:y+h, x:x+w]
        cv2.imwrite(f"debug_block_{i}_{filename}.jpg", roi)
        
        # Try processing as Numeric
        # Center Code (Block 0 usually)
        if i == 0:
            print("--- Debugging Center Code (Block 0) ---")
            debug_numeric(roi, 2, f"center_{filename}")
        
        # Set (Block 2 usually)
        if i == 2:
            print("--- Debugging Set (Block 2) ---")
            debug_set(roi, f"set_{filename}")

def debug_numeric(roi, max_cols, prefix):
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(f"debug_{prefix}_thresh.jpg", thresh)
    
    h, w = roi.shape
    num_cols = max_cols
    num_rows = 12
    col_w = w / num_cols
    row_h = h / num_rows
    
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    for c_idx in range(num_cols):
        x1 = int(c_idx * col_w)
        x2 = int((c_idx + 1) * col_w)
        
        for r_idx in range(num_rows):
            y1 = int(r_idx * row_h)
            y2 = int((r_idx + 1) * row_h)
            
            # Draw grid
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            pad_x = int(col_w * 0.25)
            pad_y = int(row_h * 0.25)
            
            if y2-pad_y > y1+pad_y and x2-pad_x > x1+pad_x:
                cell = thresh[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
                nz = cv2.countNonZero(cell)
                area = cell.size
                ratio = nz / area if area > 0 else 0
                
                if ratio > 0.1: # Log anything > 10%
                    print(f"  Col {c_idx} Row {r_idx}: Ratio={ratio:.3f}")
                    if ratio > 0.2:
                        cv2.rectangle(debug_img, (x1+pad_x, y1+pad_y), (x2-pad_x, y2-pad_y), (0, 255, 0), 2)

    cv2.imwrite(f"debug_{prefix}_grid.jpg", debug_img)

def debug_set(roi, prefix):
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(f"debug_{prefix}_thresh.jpg", thresh)
    
    h, w = roi.shape
    num_cols = 1
    num_rows = 6
    col_w = w / num_cols
    row_h = h / num_rows
    
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    for r_idx in range(num_rows):
        y1 = int(r_idx * row_h)
        y2 = int((r_idx + 1) * row_h)
        
        # Draw grid
        cv2.rectangle(debug_img, (0, y1), (w, y2), (0, 0, 255), 1)
        
        pad_x = int(col_w * 0.25)
        pad_y = int(row_h * 0.25)
        
        if y2-pad_y > y1+pad_y:
            cell = thresh[y1+pad_y:y2-pad_y, int(col_w*0.2):int(col_w*0.8)]
            nz = cv2.countNonZero(cell)
            area = cell.size
            ratio = nz / area if area > 0 else 0
            
            if ratio > 0.1:
                print(f"  Set Row {r_idx}: Ratio={ratio:.3f}")
                if ratio > 0.2:
                    cv2.rectangle(debug_img, (int(col_w*0.2), y1+pad_y), (int(col_w*0.8), y2-pad_y), (0, 255, 0), 2)

    cv2.imwrite(f"debug_{prefix}_grid.jpg", debug_img)

if __name__ == "__main__":
    debug_image("image (10).tif")
    debug_image("image (1).tif")
