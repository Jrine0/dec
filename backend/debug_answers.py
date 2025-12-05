import cv2
import numpy as np
from omr_engine import OMREngine
import os

def debug_answers(filename):
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
    
    # Extract fields logic (simplified)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    answer_contour = cnts[0]
    peri = cv2.arcLength(answer_contour, True)
    approx = cv2.approxPolyDP(answer_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
         warped_answers = engine.four_point_transform(gray, approx.reshape(4, 2))
         
         # Use engine's process_answers_block but modify it to print debug info
         # Or just call it and rely on its internal debug prints if I add them?
         # Better to copy logic here to inspect ratios.
         
         process_answers_debug(engine, warped_answers)

def process_answers_debug(engine, img):
    # ... Copy of process_answers_block logic ...
    # For brevity, I'll just use the engine's method but I'll add print statements to the engine temporarily?
    # No, better to replicate the grid sampling part.
    
    # 1. Bubble Detection
    thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)
    
    # ... (Skipping bubble detection for brevity, assuming grid logic is the key)
    
    # Let's assume we found the columns.
    # Actually, I need the columns to proceed.
    
    # Let's just use the engine's method and capture stdout?
    # Or I can modify the engine to print the ratios for Q1-Q60.
    pass

if __name__ == "__main__":
    # I will modify the engine to print ratios instead of writing a new script
    # This is faster.
    pass
