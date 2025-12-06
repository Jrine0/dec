import pandas as pd
import os

def analyze_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_results.csv")
    if not os.path.exists(csv_path):
        print("final_results.csv not found.")
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    if total_rows == 0:
        print("No data in CSV.")
        return

    # 1. Field Fill Rates
    # Check for non-empty and non-null
    center_filled = df['Center code'].notna() & (df['Center code'].astype(str).str.strip() != '')
    roll_filled = df['Roll no'].notna() & (df['Roll no'].astype(str).str.strip() != '')
    name_filled = df['Student name'].notna() & (df['Student name'].astype(str).str.strip() != '')
    set_filled = df['Set'].notna() & (df['Set'].astype(str).str.strip() != '')

    print(f"Total Sheets Processed: {total_rows}")
    print("-" * 30)
    print(f"Center Code Detected: {center_filled.sum()} ({center_filled.mean()*100:.1f}%)")
    print(f"Roll No Detected:     {roll_filled.sum()} ({roll_filled.mean()*100:.1f}%)")
    print(f"Set Detected:         {set_filled.sum()} ({set_filled.mean()*100:.1f}%)")
    print(f"Student Name Detected:{name_filled.sum()} ({name_filled.mean()*100:.1f}%)")
    print("-" * 30)

    # 2. Answer Fill Rate
    # Marks column format: "Q1:A Q2: Q3:C ..."
    total_questions = 60 # Assuming 60 questions based on previous context
    total_attempted = 0
    total_possible = total_rows * total_questions
    
    for marks_str in df['Marks']:
        if pd.isna(marks_str): continue
        
        # Split by space to get "Qx:Ans"
        parts = str(marks_str).split()
        attempted_count = 0
        for p in parts:
            if ':' in p:
                q, ans = p.split(':', 1)
                if ans.strip(): # If answer is not empty
                    attempted_count += 1
        
        total_attempted += attempted_count

    avg_attempted = total_attempted / total_rows
    fill_percentage = (total_attempted / total_possible) * 100

    print(f"Average Questions Attempted: {avg_attempted:.1f} / {total_questions}")
    print(f"Overall Answer Fill Rate:    {fill_percentage:.1f}%")

if __name__ == "__main__":
    analyze_results()
