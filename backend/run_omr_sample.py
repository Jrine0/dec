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
KEY_PATH = os.path.join(PARENT_DIR, "Untitled spreadsheet.csv")
SRS_DIR = os.path.join(PARENT_DIR, "OMR_SRS")
OUTPUT_CSV = os.path.join(BASE_DIR, "omr_results_sample.csv")

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
    q_num = str(row['Q_No'])
    section = str(row.get('Section', ''))
    option_scores = {
        "A": float(row.get('Option_A_Score', 0)),
        "B": float(row.get('Option_B_Score', 0)),
        "C": float(row.get('Option_C_Score', 0)),
        "D": float(row.get('Option_D_Score', 0))
    }
    correct_options = [opt for opt, score in option_scores.items() if score > 0]
    key_data[q_num] = {
        "section": section,
        "option_scores": option_scores,
        "correct": correct_options
    }

print(f"Loaded key for {len(key_data)} questions.")

# 2. Process Images
image_files = glob.glob(os.path.join(SRS_DIR, "*.tif"))
image_files.sort()

# Limit to 5 images
image_files = image_files[:5] 

print(f"Processing {len(image_files)} sample images...")

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
        
        # Format Marks string (Force Q1-Q60)
        marks_str = ""
        for i in range(1, 61):
            q = f"Q{i}"
            ans_list = student_answers.get(q, [])
            ans = "".join(ans_list) if ans_list else ""
            marks_str += f"{q}:{ans} "
        marks_str = marks_str.strip()

        results.append({
            "Filename": filename,
            "Center code": center_code,
            "Roll no": roll_no,
            "Set": set_code,
            "Student name": student_name,
            "Marks": marks_str,
            "Total Score": total_score
        })
        print(f"  Roll: {roll_no}, Score: {total_score}")
        
    except Exception as e:
        print(f"  Exception: {e}")

# 3. Save to CSV
if results:
    cols = ["Filename", "Center code", "Roll no", "Set", "Student name", "Marks", "Total Score"]
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, columns=cols)
    print(f"\nSUCCESS: Generated {OUTPUT_CSV}")
else:
    print("\nNo results generated.")
