import cv2
import os
import sys
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omr_engine import OMREngine

def visualize_rois():
    engine = OMREngine()
    
    # Path to a known good image
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (13).tif"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Processing {image_path}...")
    
    # 1. Load and Preprocess (Copying logic from process_sheet to get the warped image)
    image = cv2.imread(image_path)
    thresh = engine.preprocess_image(image)
    corners = engine.find_corners(thresh)
    
    if len(corners) != 4:
        print("Could not find 4 corners")
        return
        
    warped = engine.four_point_transform(image, corners)
    
    # 2. Get Contours (Copying logic from extract_fields)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    cnts = engine.get_contours(edged)
    
    # 3. Filter Large Contours (ROIs)
    large_contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 5000:
            large_contours.append(c)
    
    # Sort left-to-right to make labeling consistent
    if large_contours:
        large_contours, _ = engine.sort_contours(large_contours, method="left-to-right")
    
    # 4. Draw and Label
    vis = warped.copy()
    print(f"Found {len(large_contours)} ROIs")
    
    with open("roi_list.txt", "w") as f:
        for i, c in enumerate(large_contours):
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(vis, f"ROI {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            msg = f"ROI {i}: x={x}, y={y}, w={w}, h={h}, area={cv2.contourArea(c)}\n"
            print(msg.strip())
            f.write(msg)

        # Explore bottom half for answers
        h_img, w_img = warped.shape[:2]
        f.write("\n--- Bottom Half Candidates ---\n")
        print("\n--- Bottom Half Candidates ---")
        bottom_contours = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if y > h_img / 2 and cv2.contourArea(c) > 1000:
                bottom_contours.append(c)
        
        bottom_contours, _ = engine.sort_contours(bottom_contours, method="top-to-bottom")
        for i, c in enumerate(bottom_contours):
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            msg = f"Bottom Cand {i}: x={x}, y={y}, w={w}, h={h}, area={cv2.contourArea(c)}\n"
            print(msg.strip())
            f.write(msg)

    # Save
    output_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/roi_debug.jpg"
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_rois()
