import cv2
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omr_engine import OMREngine

def debug_q1_q5():
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (1).tif"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    engine = OMREngine()
    
    # Pre-process same as engine
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract fields (we just want the logic inside extract_fields -> process_answers_block)
    # But we can't easily call internal methods without refactoring.
    # So let's just use process_sheet and add prints inside omr_engine.py?
    # No, let's replicate the logic here for debugging.
    
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
        thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        cnts = engine.get_contours(thresh)
        bubbles = [c for c in cnts if cv2.contourArea(c) > 30]
        
        # Cluster
        bubbles, _ = engine.sort_contours(bubbles, method="left-to-right")
        
        print("Raw Bubbles (X, Area):")
        for c in bubbles[:20]: # Print first 20
            print(f"  x={cv2.boundingRect(c)[0]}, area={cv2.contourArea(c)}")
            
        columns = []
        current_col = []
        last_x = -100
        for c in bubbles:
            x, y, w, h = cv2.boundingRect(c)
            if not current_col:
                current_col.append(c)
                last_x = x
            else:
                if abs(x - last_x) < 100: 
                    current_col.append(c)
                else:
                    columns.append(current_col)
                    current_col = [c]
                    last_x = x
        if current_col:
            columns.append(current_col)
            
        print(f"Found {len(columns)} columns.")
        
        if len(columns) > 0:
            col = columns[0]
            col, _ = engine.sort_contours(col, method="top-to-bottom")
            
            # Calculate grid using Histogram Peaks
            all_cx = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]//2 for c in col]
            
            hist_range = (0, max(all_cx) + 10)
            hist_bins = int(hist_range[1] / 5)
            hist, bin_edges = np.histogram(all_cx, bins=hist_bins, range=hist_range)
            
            print(f"Histogram: {hist}")
            print(f"Bin Edges: {bin_edges}")
            
            peaks = []
            threshold = max(hist) * 0.3
            for i in range(1, len(hist)-1):
                if hist[i] > threshold and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append((bin_edges[i] + bin_edges[i+1]) / 2)
            
            if len(peaks) < 4:
                col_min_cx = np.percentile(all_cx, 5)
                col_max_cx = np.percentile(all_cx, 95)
                width_span = col_max_cx - col_min_cx
                slot_width = width_span / 3.0 if width_span > 10 else 20
                print(f"Peaks failed ({len(peaks)}). Fallback: A={col_min_cx:.1f}, D={col_max_cx:.1f}")
            else:
                peaks.sort()
                col_min_cx = peaks[0]
                col_max_cx = peaks[-1]
                width_span = col_max_cx - col_min_cx
                slot_width = width_span / 3.0 if width_span > 10 else 20
                print(f"Peaks: {peaks} -> A={col_min_cx:.1f}, D={col_max_cx:.1f}, SlotW={slot_width:.1f}")
            
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
                
            print(f"Found {len(rows)} rows.")
            
            # Analyze Q1-Q5
            for i in range(5):
                if i >= len(rows): break
                row = rows[i]
                print(f"\n--- Q{i+1} ---")
                for c in row:
                    x, y, w, h = cv2.boundingRect(c)
                    cx = x + w // 2
                    rel_x = cx - col_min_cx
                    slot_idx = int(round(rel_x / slot_width))
                    
                    ratio = engine.get_filled_bubbles(thresh, [c])[0]
                    print(f"  Bubble at x={x}, cx={cx}, rel_x={rel_x:.1f} -> Slot {slot_idx} (Ratio: {ratio:.2f})")

if __name__ == "__main__":
    debug_q1_q5()
