print("Starting script...")
import cv2
import os
import sys
import numpy as np

print("Imports done.")

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Importing OMREngine...")
from omr_engine import OMREngine
print("OMREngine imported.")

def visualize_processing():
    print("Inside visualize_processing...")
    engine = OMREngine()
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (13).tif"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Processing {image_path}...")
    image = cv2.imread(image_path)
    
    # Use Canny as it seems to find edges well
    print("Using Canny for contour visualization")
    edged = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 75, 200)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    debug_corners_img = image.copy()
    
    # Draw top 20 contours
    print(f"Total contours found: {len(cnts)}")
    for i, c in enumerate(cnts[:20]):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # Color cycle
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        color = colors[i % len(colors)]
        
        cv2.drawContours(debug_corners_img, [c], -1, color, 3)
        cv2.putText(debug_corners_img, f"#{i} ({int(area)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"Contour #{i}: Area={area}, Rect=(x={x}, y={y}, w={w}, h={h})")

    cv2.imwrite("debug_all_contours.jpg", debug_corners_img)
    print("Saved debug_all_contours.jpg")
    
    return

if __name__ == '__main__':
    visualize_processing()
