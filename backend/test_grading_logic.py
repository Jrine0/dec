
import pandas as pd
import os

# Mock Scorer from scoring.py (mimic behavior)
class Scorer:
    def calculate_score(self, student_answers, answer_key):
        total_score = 0
        details = {}
        for q_num, key_data in answer_key.items():
            option_scores = key_data["option_scores"]
            student_selection = set(student_answers.get(q_num, []))
            
            score = 0
            if student_selection:
                 # Simplified for test: single choice match
                 # Assuming single correct answer for simplicity or full match
                 score = sum(option_scores.get(opt, 0) for opt in student_selection)
            
            total_score += score
            details[q_num] = {"score": score}
        return total_score, details

# Test Data Setup
def test_grading():
    # 1. Mock Answer Key (DataFrame) simulating CSV
    # Row 1-10: Section 1
    # Row 31-60: Section 2
    # Others: Section 3 or None
    
    data = {
        'Q_No': ['1', '2', '10', '11', '31', '40', '60', 'Q1'], # Test format mix
        'Section': ['S1', 'S1', 'S1', 'SOther', 'S2', 'S2', 'S2', 'S1'],
        'Option_A_Score': [1, 1, 1, 1, 1, 1, 1, 1],
        'Option_B_Score': [0, 0, 0, 0, 0, 0, 0, 0],
        'Option_C_Score': [0, 0, 0, 0, 0, 0, 0, 0],
        'Option_D_Score': [0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Logic extracted from main.py (Key Normalization)
    key_data = {}
    for _, row in df.iterrows():
        q_num_raw = str(row['Q_No']).strip()
        if not q_num_raw.lower().startswith('q'):
            q_num = f"Q{q_num_raw}"
        else:
            q_num = q_num_raw.capitalize()

        key_data[q_num] = {
            "option_scores": {
                "A": float(row['Option_A_Score']), 
                "B": 0, "C": 0, "D": 0
            }
        }
        
    print(f"DEBUG: Normalized Keys: {list(key_data.keys())}")
    
    # 2. Mock Student Answers
    # Student answers correctly A for everything
    student_answers = {
        "Q1": ["A"], "Q2": ["A"], "Q10": ["A"], # S1: 3 marks
        "Q11": ["A"], # Other: 1 mark
        "Q31": ["A"], "Q40": ["A"], "Q60": ["A"] # S2: 3 marks
        # Total Real Score: 7
    }
    
    scorer = Scorer()
    total_score, details = scorer.calculate_score(student_answers, key_data)
    
    # 3. Test Section Logic (copied from main.py)
    section_1_score = 0
    section_2_score = 0
    
    for q, det in details.items():
        try:
            q_int = int(q.replace('Q', ''))
            if 1 <= q_int <= 30:
                section_1_score += det['score']
            elif 31 <= q_int <= 60:
                section_2_score += det['score']
        except:
            pass
            
    final_marks = section_1_score + section_2_score
    
    print(f"Total Score (All): {total_score}")
    print(f"Section 1 Score (Expected 4): {section_1_score}")
    print(f"Section 2 Score (Expected 3): {section_2_score}")
    print(f"Final Marks (S1+S2): {final_marks}")
    
    assert section_1_score == 4
    assert section_2_score == 3
    assert final_marks == 7
    print("TEST PASSED")

if __name__ == "__main__":
    test_grading()
