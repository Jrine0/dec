import cv2
from omr_engine import OMREngine
import os

# Path to the uploaded image (adjust as needed)
image_path = "C:/Users/jitin/.gemini/antigravity/brain/411d7566-e88f-47b2-a396-12ec9ce76a20/uploaded_image_1764613869651.png"

if not os.path.exists(image_path):
    print(f"Image not found at {image_path}")
    exit()

engine = OMREngine()
try:
    print("Processing image...")
    results = engine.process_sheet(image_path)
    print("Results Keys:", results.keys())
    print("Roll No:", results.get("roll_no"))
    print("Answers Found:", len(results.get("answers", {})))
    print("First 5 Answers:", list(results.get("answers", {}).items())[:5])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
