from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from typing import List
from fastapi import File, UploadFile, BackgroundTasks
from omr_engine import OMREngine
from scoring import Scorer
from utils import pdf_to_images
import pandas as pd
import cv2

app = FastAPI(title="OMR Processing API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

omr_engine = OMREngine()
scorer = Scorer()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
    return {"message": f"Uploaded {len(saved_files)} files", "files": saved_files}

@app.post("/process")
def process_omr(answer_key: UploadFile = File(...)):
    try:
        # 1. Load Answer Key
        key_path = os.path.join(UPLOAD_DIR, answer_key.filename)
        with open(key_path, "wb") as buffer:
            shutil.copyfileobj(answer_key.file, buffer)
        
        # Parse CSV with smart header detection
        # First, read without header to find the row with "Q_No"
        temp_df = pd.read_csv(key_path, header=None, encoding='utf-8-sig')
        
        header_row_index = 0
        found_header = False
        
        for i, row in temp_df.iterrows():
            # Check if this row contains "Q_No" (case insensitive)
            row_values = [str(val).strip() for val in row.values]
            if "Q_No" in row_values:
                header_row_index = i
                found_header = True
                break
        
        if found_header:
            df = pd.read_csv(key_path, header=header_row_index, encoding='utf-8-sig')
        else:
            # Fallback to default if not found (will likely fail later but we tried)
            df = pd.read_csv(key_path, encoding='utf-8-sig')

        # Strip whitespace from column names to avoid KeyErrors
        df.columns = df.columns.str.strip()
        
        print(f"DEBUG: Found header at row {header_row_index}")
        print(f"DEBUG: CSV Columns: {df.columns.tolist()}")
        print(f"DEBUG: First row: {df.iloc[0].to_dict()}")
        
        # Log to file for debugging
        with open("backend_error.log", "a") as f:
            f.write(f"CSV Columns: {df.columns.tolist()}\n")
            f.write(f"First Row: {df.iloc[0].to_dict()}\n")
        
        key_data = {} # Initialize key_data
        
        for _, row in df.iterrows():
            q_num_raw = str(row['Q_No']).strip()
            # Normalize to Q1, Q2...
            if not q_num_raw.lower().startswith('q'):
                q_num = f"Q{q_num_raw}"
            else:
                q_num = q_num_raw.capitalize() # ensure Q1 not q1

            section = str(row.get('Section', ''))
            
            # Extract scores for each option

            raw_scores = {
                "A": float(row.get('Option_A_Score', 0)),
                "B": float(row.get('Option_B_Score', 0)),
                "C": float(row.get('Option_C_Score', 0)),
                "D": float(row.get('Option_D_Score', 0))
            }
            # Force Correct Answer to 4.0, others to 0.0
            max_val = max(raw_scores.values())
            option_scores = {k: (4.0 if v == max_val and v > 0 else 0.0) for k, v in raw_scores.items()}
            
            # Determine correct options (those with positive score)
            correct_options = [opt for opt, score in option_scores.items() if score > 0]
            
            key_data[q_num] = {
                "section": section,
                "option_scores": option_scores,
                "correct": correct_options
            }

        results = []
        
        # 2. Process Images
        all_files = os.listdir(UPLOAD_DIR)
        
        for filename in all_files:
            if filename == answer_key.filename: continue
            
            file_path = os.path.join(UPLOAD_DIR, filename)
            ext = filename.split('.')[-1].lower()
            
            images_to_process = []
            
            if ext == 'pdf':
                images_to_process = pdf_to_images(file_path)
            elif ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
                images_to_process = [cv2.imread(file_path)]
            else:
                print(f"Skipping unsupported file type: {filename}")
                
            for img in images_to_process:
                 try:
                     # Process
                     extraction_result = omr_engine.process_sheet(img)
                     
                     if "error" in extraction_result:
                         print(f"Error processing {filename}: {extraction_result['error']}")
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
                     
                     # Format Marks string (e.g. "1:A 2:B ...")
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
                         "filename": filename,
                         "center_code": center_code,
                         "roll_no": roll_no,
                         "set": set_code,
                         "student_name": student_name,
                         "marks": marks_str,
                         "total_score": total_score,
                         "section_1_score": section_1_score,
                         "section_2_score": section_2_score,
                         "final_marks": final_marks,
                         "details": details
                     })
                 except Exception as e:
                     print(f"Exception processing {filename}: {e}")
                     import traceback
                     with open("backend_error.log", "a") as f:
                         f.write(f"Error processing {filename}: {str(e)}\n")
                         f.write(traceback.format_exc())
                         f.write("\n")
                     
        # Save results to CSV
        if results:
            output_csv = os.path.join(RESULTS_DIR, "results.csv")
            # Flatten for CSV
            csv_data = []
            for r in results:
                row = {
                    "Filename": r["filename"],
                    "Center code": r["center_code"],
                    "Roll no": r["roll_no"],
                    "Set": r["set"],
                    "Student name": r["student_name"],
                    "Marks": r["marks"],
                    "Section 1": r["section_1_score"],
                    "Section 2": r["section_2_score"],
                    "Final Marks": r["final_marks"]
                }
                csv_data.append(row)
                
            # Define column order
            cols = ["Filename", "Center code", "Roll no", "Set", "Student name", "Marks", "Section 1", "Section 2", "Final Marks"]
            pd.DataFrame(csv_data).to_csv(output_csv, index=False, columns=cols)
            return {"message": "Processing complete", "results_file": output_csv, "data": results}
        
        return {"message": "No results generated", "status": "failed"}

    except Exception as e:
        import traceback
        error_msg = f"Global error in process_omr: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("backend_error.log", "a") as f:
            f.write(error_msg)
        
        # Include columns in error message if possible
        cols = "Unknown"
        try:
            cols = str(df.columns.tolist())
        except:
            pass
            
        return {"message": f"Server Error: {str(e)}. Found columns: {cols}", "status": "error"}
