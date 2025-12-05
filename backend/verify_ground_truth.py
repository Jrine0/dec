import cv2
import numpy as np
import sys
import os

# Add current directory to path to import backend modules
sys.path.append(os.getcwd())

from omr_engine import OMREngine

GROUND_TRUTH_STR = "B D B A D D C C B D B D A A A C A A D C D A A A A B A A D C B A B A A C A A B A C A C A C A B A B A C C B A A B B A A A"
GROUND_TRUTH = GROUND_TRUTH_STR.split()

def verify_image():
    engine = OMREngine()
    image_path = os.path.join("..", "OMR_SRS", "image (1).tif")
    
    print(f"Processing {image_path}...")
    
    # We need to capture the raw ratios, so we might need to modify omr_engine again 
    # or just rely on the final answer and print mismatches.
    # Ideally, we want to see the ratios for mismatches.
    # Let's assume omr_engine is currently clean (no debug prints).
    # We will run it and get the 'answers' dict.
    
    try:
        results = engine.process_sheet(image_path)
        answers = results.get("answers", {})
        
        correct_count = 0
        total_count = 60
        
        print("\n--- Verification Report ---")
        print(f"{'Q#':<4} {'Exp':<4} {'Got':<4} {'Status':<10}")
        print("-" * 30)
        
        mismatches = []
        
        for i in range(total_count):
            q_num = i + 1
            expected = GROUND_TRUTH[i]
            
            # Get detected answer
            # answers key is "Q1", "Q2"... value is list ["A"] or []
            detected_list = answers.get(f"Q{q_num}", [])
            detected = detected_list[0] if detected_list else ""
            
            is_match = (detected == expected)
            if is_match:
                correct_count += 1
                status = "OK"
            else:
                status = "FAIL"
                mismatches.append(q_num)
            
            print(f"Q{q_num:<3} {expected:<4} {detected:<4} {status}")
            
        accuracy = (correct_count / total_count) * 100
        print("-" * 30)
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        print(f"Mismatches: {mismatches}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_image()
