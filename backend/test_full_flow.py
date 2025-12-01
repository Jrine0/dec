import urllib.request
import urllib.parse
import os
import mimetypes
import uuid
import glob
import json
import csv

def upload_files(url, key_path, image_paths):
    boundary = uuid.uuid4().hex
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    }
    
    data = []
    
    # Add Answer Key
    filename = os.path.basename(key_path)
    mime_type = 'text/csv'
    data.append(f'--{boundary}'.encode())
    data.append(f'Content-Disposition: form-data; name="answer_key"; filename="{filename}"'.encode())
    data.append(f'Content-Type: {mime_type}'.encode())
    data.append(b'')
    with open(key_path, 'rb') as f:
        data.append(f.read())
    data.append(b'')
    
    # Add Images
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        mime_type = mimetypes.guess_type(img_path)[0] or 'application/octet-stream'
        data.append(f'--{boundary}'.encode())
        data.append(f'Content-Disposition: form-data; name="files"; filename="{filename}"'.encode())
        data.append(f'Content-Type: {mime_type}'.encode())
        data.append(b'')
        with open(img_path, 'rb') as f:
            data.append(f.read())
        data.append(b'')
        
    data.append(f'--{boundary}--'.encode())
    data.append(b'')
    
    body = b'\r\n'.join(data)
    
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            return response.read().decode('utf-8'), response.status
    except urllib.error.HTTPError as e:
        return e.read().decode('utf-8'), e.code
    except Exception as e:
        return str(e), 0

# Configuration
URL = 'http://localhost:8000/process'
# Use the user's specific file
KEY_PATH = r'c:/Users/jitin/Desktop/Desk/drive/Documents/omr/Untitled spreadsheet.csv'
SRS_DIR = r'c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS'
OUTPUT_CSV = 'final_results.csv'

# Get first 5 TIF images
image_files = glob.glob(os.path.join(SRS_DIR, "*.tif"))[:5]

print(f"Testing with key: {os.path.basename(KEY_PATH)}")
print(f"Processing {len(image_files)} images...")

resp, status = upload_files(URL, KEY_PATH, image_files)

if status == 200:
    try:
        response_json = json.loads(resp)
        results = response_json.get('data', [])
        
        if not results:
             print("WARNING: No results returned in data field.")
             print(f"Full response: {resp}")
        
        # Generate CSV
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Roll No", "Total Score"])
            
            for r in results:
                if isinstance(r, dict):
                    writer.writerow([r.get('filename'), r.get('roll_no'), r.get('total_score')])
                else:
                    print(f"Skipping invalid result: {r}")
                
        print(f"\nSUCCESS: Generated {OUTPUT_CSV}")
        
        # Read and print the CSV content
        print("\n--- CSV Content ---")
        with open(OUTPUT_CSV, 'r') as f:
            print(f.read())
        print("-------------------")
        
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        print(resp)
else:
    print(f"Failed with status {status}")
    print(resp)
