import cv2
import os
import glob
import pandas as pd
import numpy as np
from omr_engine import OMREngine
from scoring import Scorer

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRS_DIR = os.path.join(os.path.dirname(BASE_DIR), "OMR_SRS")
ANSWER_KEY_CSV = os.path.join(os.path.dirname(BASE_DIR), "answers.csv")

def debug_single_image(image_name):
    # Load Answer Key
    df = pd.read_csv(ANSWER_KEY_CSV, header=2)
    key_data = {}
    for _, row in df.iterrows():
        q_num = str(row.get('Q_No'))
        if q_num.lower().startswith('q'):
            q_num = q_num.capitalize()
        else:
            q_num = f"Q{q_num}"
        
        section = str(row.get('Section', ''))
        raw_scores = {
            "A": float(row.get('Option_A_Score', 0)),
            "B": float(row.get('Option_B_Score', 0)),
            "C": float(row.get('Option_C_Score', 0)),
            "D": float(row.get('Option_D_Score', 0))
        }
        max_val = max(raw_scores.values())
        option_scores = {k: (4.0 if v == max_val and v > 0 else 0.0) for k, v in raw_scores.items()}
        
        correct_options = [opt for opt, score in option_scores.items() if score > 0]
        key_data[q_num] = {
            "section": section,
            "option_scores": option_scores,
            "correct": correct_options
        }

    engine = OMREngine()
    scorer = Scorer()

    img_path = os.path.join(SRS_DIR, image_name)
    print(f"DEBUG: Processing {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found")
        return

    # Process
    result = engine.process_sheet(img)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    answers = result.get("answers", {})
    print(f"DEBUG: Detected Answers: {len(answers)}")
    
    # Detailed Score Check
    total_score, details = scorer.calculate_score(answers, key_data)
    
    print("\n--- Detailed Q-by-Q Analysis ---")
    keys_sorted = sorted(key_data.keys(), key=lambda x: int(x.replace('Q','')))
    
    for q in keys_sorted:
        student_ans = "".join(answers.get(q, []))
        correct_ans = "".join(key_data[q]['correct'])
        match = student_ans == correct_ans
        
        # Only print if mismatch or empty
        # if not match or not student_ans:
        print(f"{q}: Student='{student_ans}' | Key='{correct_ans}' | Match={match}")

if __name__ == "__main__":
    debug_single_image("image (1).tif")
