import os
import cv2
import pandas as pd
import glob
from omr_engine import OMREngine
from scoring import Scorer
from utils import pdf_to_images
import sys

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
KEY_PATH = os.path.join(PARENT_DIR, "answers.csv")
SRS_DIR = os.path.join(PARENT_DIR, "OMR_SRS")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_results.csv")

print(f"Key Path: {KEY_PATH}")
print(f"Images Dir: {SRS_DIR}")
print(f"Output CSV: {OUTPUT_CSV}")

# Initialize
engine = OMREngine()
scorer = Scorer()

# 1. Load Answer Key
if not os.path.exists(KEY_PATH):
    print(f"ERROR: Key file not found at {KEY_PATH}")
    sys.exit(1)

# Parse CSV with smart header detection
temp_df = pd.read_csv(KEY_PATH, header=None, encoding='utf-8-sig')
header_row_index = 0
found_header = False

for i, row in temp_df.iterrows():
    row_values = [str(val).strip() for val in row.values]
    if "Q_No" in row_values:
        header_row_index = i
        found_header = True
        break

if found_header:
    df = pd.read_csv(KEY_PATH, header=header_row_index, encoding='utf-8-sig')
else:
    df = pd.read_csv(KEY_PATH, encoding='utf-8-sig')

df.columns = df.columns.str.strip()
print(f"Key Columns: {df.columns.tolist()}")

key_data = {}
for _, row in df.iterrows():
    q_num_raw = str(row['Q_No']).strip()
    # Normalize to Q1, Q2...
    if not q_num_raw.lower().startswith('q'):
        q_num = f"Q{q_num_raw}"
    else:
        q_num = q_num_raw.capitalize()

    section = str(row.get('Section', ''))
    raw_scores = {
        "A": float(row.get('Option_A_Score', 0)),
        "B": float(row.get('Option_B_Score', 0)),
        "C": float(row.get('Option_C_Score', 0)),
        "D": float(row.get('Option_D_Score', 0))
    }
    # Force Correct Answer to 4.0, others to 0.0
    max_val = max(raw_scores.values())
    option_scores = {k: (4.0 if v == max_val and v > 0 else 0.0) for k, v in raw_scores.items()}

    correct_options = [opt for opt, score in option_scores.items() if score > 0]
    key_data[q_num] = {
        "section": section,
        "option_scores": option_scores,
        "correct": correct_options
    }

print(f"Loaded key for {len(key_data)} questions.")

# 2. Process Images
image_files = glob.glob(os.path.join(SRS_DIR, "*.tif"))
# Sort to process in order
image_files.sort()

# Limit for testing if needed, but user asked for "all marks"
image_files = image_files # Process all 

print(f"Found {len(image_files)} images to process.")

results = []

for i, file_path in enumerate(image_files):
    filename = os.path.basename(file_path)
    print(f"[{i+1}/{len(image_files)}] Processing {filename}...")
    
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"  Failed to load image: {file_path}")
            continue
            
        extraction_result = engine.process_sheet(img)
        
        if "error" in extraction_result:
            print(f"  Error: {extraction_result['error']}")
            continue
            
        student_answers = extraction_result.get("answers", {})
        roll_no = extraction_result.get("roll_no", "")
        center_code = extraction_result.get("center_code", "")
        set_code = extraction_result.get("set", "")
        student_name = extraction_result.get("student_name", "")
        
        # Score
        total_score, details = scorer.calculate_score(student_answers, key_data)
        
        # --- Calculate Section Scores ---
        section_1_score = 0
        section_2_score = 0
        
        for q, det in details.items():
            # Extract numeric part from "Q1" -> 1
            try:
                q_int = int(q.replace('Q', ''))
                
                if 1 <= q_int <= 30:
                    section_1_score += det['score']
                elif 31 <= q_int <= 60:
                    section_2_score += det['score']
            except:
                pass
                
        final_marks = section_1_score + section_2_score

        # Format Marks string
        marks_str = ""
        try:
            sorted_qs = sorted(student_answers.keys(), key=lambda x: int(x.replace('Q', '')))
        except:
            sorted_qs = sorted(student_answers.keys())
            
        for q in sorted_qs:
            ans = "".join(student_answers[q])
            marks_str += f"{q}:{ans} "
        marks_str = marks_str.strip()

        results.append({
            "Filename": filename,
            "Center code": center_code,
            "Roll no": roll_no,
            "Set": set_code,
            "Student name": student_name,
            "Marks": marks_str,
            "Section 1": section_1_score,
            "Section 2": section_2_score,
            "Final Marks": final_marks
        })
        print(f"  Roll: {roll_no}, Final: {final_marks}, Score: {total_score}")
        
    except Exception as e:
        print(f"  Exception: {e}")
        # import traceback
        # traceback.print_exc()

# 3. Save to CSV
if results:
    cols = ["Filename", "Center code", "Roll no", "Set", "Student name", "Marks", "Section 1", "Section 2", "Final Marks"]
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, columns=cols)
    print(f"\nSUCCESS: Generated {OUTPUT_CSV}")
    print(f"Total processed: {len(results)}")
else:
    print("\nNo results generated.")
