import urllib.request
import urllib.parse
import os
import mimetypes
import uuid
import glob

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
KEY_PATH = r'c:/Users/jitin/Desktop/Desk/drive/Documents/omr/sample_key.csv'
SRS_DIR = r'c:/Users/jitin/Desktop/Desk/drive/Documents/omr/OMR_SRS'

# Get first 5 TIF images
image_files = glob.glob(os.path.join(SRS_DIR, "*.tif"))[:5]

print(f"Testing with {len(image_files)} images from {SRS_DIR}...")
for img in image_files:
    print(f" - {os.path.basename(img)}")

resp, status = upload_files(URL, KEY_PATH, image_files)

print(f"\nStatus: {status}")
print(f"Response: {resp[:500]}...")

if status == 200 and "results" in resp:
    print("\nSUCCESS: Backend processed OMR_SRS images!")
else:
    print("\nFAILURE: Backend failed to process images.")
