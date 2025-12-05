import cv2
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omr_engine import OMREngine

def test_omr():
    engine = OMREngine()
    
    # Path to an image
    image_path = r"c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS/image (13).tif"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Processing {image_path}...")
    
    try:
        results = engine.process_sheet(image_path)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\n--- Extraction Results ---")
            print(f"Center Code: {results.get('center_code')}")
            print(f"Roll No: {results.get('roll_no')}")
            print(f"Set: {results.get('set')}")
            print(f"Student Name: {results.get('student_name')}")
            print(f"Answers Found: {len(results.get('answers', {}))}")
            
            # Print first few answers
            answers = results.get('answers', {})
            try:
                sorted_keys = sorted(answers.keys(), key=lambda x: int(x.replace('Q', '')))
            except:
                sorted_keys = sorted(answers.keys())
                
            print("First 5 Answers:")
            for k in sorted_keys[:5]:
                print(f"  {k}: {answers[k]}")
                
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_omr()
