import os
import cv2
import pandas as pd
from omr_engine import OMREngine
from scoring import Scorer
from utils import pdf_to_images

import sys

# Redirect stdout to file
sys.stdout = open("debug_log.txt", "w")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
KEY_PATH = os.path.join(PARENT_DIR, "sample_key.csv")
FILES = ["sample 1.pdf", "sample 3.pdf"]

# Initialize
engine = OMREngine()
scorer = Scorer()

# Load Key
print(f"Loading key from {KEY_PATH}...")
df = pd.read_csv(KEY_PATH)
key_data = {}
for _, row in df.iterrows():
    q_num = str(row['Q_No'])
    section = str(row.get('Section', ''))
    option_scores = {
        "A": float(row.get('Option_A_Score', 0)),
        "B": float(row.get('Option_B_Score', 0)),
        "C": float(row.get('Option_C_Score', 0)),
        "D": float(row.get('Option_D_Score', 0))
    }
    correct_options = [opt for opt, score in option_scores.items() if score > 0]
    key_data[q_num] = {"section": section, "option_scores": option_scores, "correct": correct_options}

print("Key loaded.")

# Process Files
for filename in FILES:
    file_path = os.path.join(PARENT_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    print(f"\nProcessing {filename}...")
    try:
        # Convert PDF to images
        images = pdf_to_images(file_path)
        print(f"  Converted to {len(images)} images.")
        
        for i, img in enumerate(images):
            print(f"  Processing page {i+1}...")
            
            # Run OMR
            results = engine.process_sheet(img)
            
            if "error" in results:
                print(f"  ERROR: {results['error']}")
                continue
                
            answers = results["answers"]
            roll_no = results["roll_no"]
            print(f"  Roll No: {roll_no}")
            print(f"  Answers Detected: {len(answers)}")
            
            # Calculate Score
            total_score, details = scorer.calculate_score(answers, key_data)
            print(f"  Total Score: {total_score}")
            
            # Print first few details
            print("  Sample Details:")
            for q in list(details.keys())[:5]:
                print(f"    Q{q}: Selected {details[q]['selected']} -> Score {details[q]['score']} ({details[q]['status']})")
                
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
