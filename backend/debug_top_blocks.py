import cv2
import numpy as np
from omr_engine import OMREngine

def debug_top_blocks():
    engine = OMREngine()
    img_path = r"c:\Users\jitin\Desktop\Desk\drive\Documents\omr\OMR_SRS\image (1).tif"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge Detection to find blocks (Copying logic from extract_fields)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Filter for top blocks
    top_blocks = []
    for c in cnts[2:]:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if 100000 < area < 200000 and y < 1000:
            top_blocks.append(c)
            
    top_blocks, _ = engine.sort_contours(top_blocks, method="left-to-right")
    
    if len(top_blocks) >= 3:
        center_contour = top_blocks[0]
        roll_contour = top_blocks[1]
        set_contour = top_blocks[2] # Assuming Set is the 3rd block
        
        # Debug Center Code
        debug_block(engine, gray, center_contour, "Center Code", 2, 12) 
        
        # Debug Roll No
        debug_block(engine, gray, roll_contour, "Roll No", 4, 12) 
        
        # Debug Set
        debug_block(engine, gray, set_contour, "Set", 1, 6)
    else:
        print("Could not find all top blocks")

def debug_block(engine, gray, contour, name, num_cols, num_rows):
    x, y, w, h = cv2.boundingRect(contour)
    print(f"\n--- Debugging {name} ---")
    print(f"ROI: {x},{y},{w},{h}")
    
    roi = gray[y:y+h, x:x+w]
    cv2.imwrite(f"debug_{name.lower().replace(' ', '_')}_raw.jpg", roi)
    
    # Grid Sampling Logic
    col_w = w / num_cols
    row_h = h / num_rows
    
    print(f"Grid: {num_cols}x{num_rows}, Cell: {col_w:.1f}x{row_h:.1f}")
    
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"debug_{name.lower().replace(' ', '_')}_thresh.jpg", thresh)
    
    # Draw Grid
    debug_grid = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for c_idx in range(num_cols + 1):
        cx = int(c_idx * col_w)
        cv2.line(debug_grid, (cx, 0), (cx, int(h)), (0, 255, 0), 1)
    for r_idx in range(num_rows + 1):
        cy = int(r_idx * row_h)
        cv2.line(debug_grid, (0, cy), (int(w), cy), (0, 0, 255), 1)
    cv2.imwrite(f"debug_{name.lower().replace(' ', '_')}_grid.jpg", debug_grid)
    
    result = ""
    for c_idx in range(num_cols):
        x1 = int(c_idx * col_w)
        x2 = int((c_idx + 1) * col_w)
        
        best_r_idx = -1
        max_ratio = 0.0
        
        for r_idx in range(num_rows):
            y1 = int(r_idx * row_h)
            y2 = int((r_idx + 1) * row_h)
            
            pad_x = int(col_w * 0.25)
            pad_y = int(row_h * 0.25)
            
            if y2-pad_y > y1+pad_y and x2-pad_x > x1+pad_x:
                cell = thresh[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
                nz = cv2.countNonZero(cell)
                area = cell.size
                if area > 0:
                    ratio = nz / area
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_r_idx = r_idx
        
        print(f"Col {c_idx}: Max Ratio {max_ratio:.2f} at Row {best_r_idx}")
        
        # Map Row to Digit (Row 2 -> 0)
        val_idx = best_r_idx - 2
        
        if max_ratio > 0.2: # Very Lower Threshold
             if val_idx >= 0:
                 result += str(val_idx)
             else:
                 result += "H" # Header
        else:
             result += "?"
             
    print(f"Result: {result}")

if __name__ == "__main__":
    debug_top_blocks()
