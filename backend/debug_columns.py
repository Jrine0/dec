import cv2
import numpy as np
import os
from omr_engine import OMREngine

engine = OMREngine()
img_path = r"C:\Users\jitin\Desktop\Desk\drive\Documents\omr\OMR_SRS\image (1).tif"
img = cv2.imread(img_path)

# Extract answers ROI like in process_sheet
# We need to minimally replicate the step to get the warped image
# For improved debugging, we will just call process_sheet but hook into the internals? 
# No, easier to copy-paste the extraction part or modify omr_engine to export debug data.

# Let's modify omr_engine to simply print the PEAKS for the first column.
print("Analyzing column peaks...")

# Manually running extraction steps to get the Warped ROI
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
cnts = engine.get_contours(thresh)
# Filter for answer block (largest rect roughly)
# ... this is complex to replicate reliably without importing logic.

# Alternative: We already suspect the column logic in omr_engine.py.
# Let's write a script that runs engine.process_sheet() and allows the engine to print detailed column debug info.
# I will modify omr_engine.py to print the "peaks" list and "cols_x" list for Col 0.

result = engine.process_sheet(img)
