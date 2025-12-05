import cv2
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omr_engine import OMREngine

def analyze_debug_image():
    image_path = "debug_warped_answers.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    
    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    engine = OMREngine()
    cnts = engine.get_contours(thresh)
    
    bubbles = []
    for c in cnts:
        if cv2.contourArea(c) > 30:
            bubbles.append(c)
            
    print(f"Found {len(bubbles)} bubbles.")
    
    # Sort left-to-right to find columns
    bubbles, _ = engine.sort_contours(bubbles, method="left-to-right")
    
    columns = []
    current_col = []
    last_x = -100
    
    for c in bubbles:
        x, y, w, h = cv2.boundingRect(c)
        if not current_col:
            current_col.append(c)
            last_x = x
        else:
            if abs(x - last_x) < 40: 
                current_col.append(c)
            else:
                columns.append(current_col)
                current_col = [c]
                last_x = x
    if current_col:
        columns.append(current_col)
        
    print(f"Found {len(columns)} columns.")
    
    # Analyze Column 0 (Q1-Q15)
    if len(columns) > 0:
        col = columns[0]
        col, _ = engine.sort_contours(col, method="top-to-bottom")
        
        rows = []
        current_row = []
        last_y = -100
        
        heights = [cv2.boundingRect(c)[3] for c in col]
        avg_h = sum(heights) / len(heights)
        
        for c in col:
            x, y, w, h = cv2.boundingRect(c)
            if not current_row:
                current_row.append(c)
                last_y = y
            else:
                if abs(y - last_y) < avg_h * 0.9:
                    current_row.append(c)
                else:
                    rows.append(current_row)
                    current_row = [c]
                    last_y = y
        if current_row:
            rows.append(current_row)
            
        print(f"Column 0 has {len(rows)} rows.")
        
        # Check Q1 (Row 0)
        if len(rows) > 0:
            row = rows[0]
            row, _ = engine.sort_contours(row, method="left-to-right")
            print(f"Q1 has {len(row)} bubbles.")
            
            ratios = engine.get_filled_bubbles(thresh, row)
            options = ['A', 'B', 'C', 'D']
            
            print("Q1 Fill Ratios:")
            for i, r in enumerate(ratios):
                opt = options[i] if i < 4 else "?"
                print(f"  {opt}: {r:.4f}")
                
            max_idx = np.argmax(ratios)
            print(f"  Max: {options[max_idx] if max_idx < 4 else '?'} ({ratios[max_idx]:.4f})")

if __name__ == "__main__":
    analyze_debug_image()
