import os
import cv2
from utils import pdf_to_images
import numpy as np

# Create a dummy PDF for testing if one doesn't exist
# Actually, let's just mock the PDF part or assume the user will upload one.
# Since I don't have a PDF, I will create a dummy image and save it as PDF using PIL to test the conversion back.

from PIL import Image

def create_dummy_pdf(path):
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(path)

pdf_path = "test_doc.pdf"
create_dummy_pdf(pdf_path)

try:
    print(f"Testing PDF conversion for {pdf_path}...")
    images = pdf_to_images(pdf_path)
    print(f"Converted {len(images)} images.")
    
    if len(images) > 0:
        print(f"First image shape: {images[0].shape}")
        # Verify it's a valid image
        if images[0].shape == (200, 200, 3):
            print("SUCCESS: Image shape matches (scaled).")
        else:
            print(f"WARNING: Image shape {images[0].shape} != (200, 200, 3), but conversion successful.")
    else:
        print("FAILURE: No images extracted.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as e:
            print(f"Warning: Could not remove {pdf_path}: {e}")
