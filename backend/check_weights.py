import pandas as pd

def check_weights():
    try:
        df = pd.read_csv('../answers.csv', header=2)
        print("Loaded answers.csv")
        print(f"Columns: {df.columns.tolist()}")
        
        odd_weights = []
        for index, row in df.iterrows():
            q_no = row.get('Q_No')
            for col in ['Option_A_Score', 'Option_B_Score', 'Option_C_Score', 'Option_D_Score']:
                val = row.get(col)
                try:
                    score = float(val)
                    if score != 0 and score != 4:
                        odd_weights.append(f"Q{q_no} - {col}: {score}")
                except Exception as e:
                    pass
                    # print(f"Error parsing {col} for Q{q_no}: {val}")

        if odd_weights:
            print("FOUND STRANGE WEIGHTS:")
            for w in odd_weights:
                print(w)
        else:
            print("All weights are either 0 or 4.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_weights()
