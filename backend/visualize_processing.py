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
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (1).tif"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Processing {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return
        
    print(f"Image Shape: {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Gray Stats: min={np.min(gray)}, max={np.max(gray)}, mean={np.mean(gray)}")
    
    print(f"Gray Stats: min={np.min(gray)}, max={np.max(gray)}, mean={np.mean(gray)}")
    
    # Run the full process to generate debug images (like debug_warped_answers.jpg)
    print("Running engine.process_sheet()...")
    try:
        results = engine.process_sheet(image)
        print("Process sheet finished.")
        print(f"Results: {results.get('answers', 'No answers')}")
    except Exception as e:
        print(f"Error in process_sheet: {e}")

    # Now we can analyze the *result* of that, but for now let's just stop here 
    # as process_sheet saves 'debug_warped_answers.jpg'
    return

if __name__ == '__main__':
    visualize_processing()
