class Scorer:
    def __init__(self):
        pass

    def calculate_score(self, student_answers, answer_key):
        # student_answers: dict {question_num: [selected_options]}
        # answer_key: pandas DataFrame or dict {question_num: {"correct": [options], "marks": float}}
        
        total_score = 0
        details = {}

    def calculate_score(self, student_answers, answer_key):
        # student_answers: dict {question_num: [selected_options]}
        # answer_key: dict {question_num: {"option_scores": {A:1, B:0...}, "correct": [A]}}
        
        total_score = 0
        details = {}

        for q_num, key_data in answer_key.items():
            section = key_data.get("section", "")
            option_scores = key_data["option_scores"]
            correct_options = set(key_data["correct"])
            
            student_selection = set(student_answers.get(q_num, []))
            
            score = 0
            status = "unanswered"

            if not student_selection:
                score = 0
                status = "unanswered"
            else:
                # Check for any wrong option
                # A wrong option is one that is NOT in the correct_options set (or has 0 score)
                has_wrong = any(opt not in correct_options for opt in student_selection)
                
                if has_wrong:
                    score = 0
                    status = "wrong"
                else:
                    # All selected are correct (subset or full match)
                    # Sum the scores of the selected options
                    score = sum(option_scores.get(opt, 0) for opt in student_selection)
                    
                    if student_selection == correct_options:
                        status = "correct"
                    else:
                        status = "partial"
            
            total_score += score
            details[q_num] = {
                "section": section,
                "selected": list(student_selection),
                "correct": list(correct_options),
                "score": score,
                "status": status
            }
            
        return total_score, details
            
        return total_score, details
