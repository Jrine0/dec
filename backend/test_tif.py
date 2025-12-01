import urllib.request
import urllib.parse
import os
import json
import mimetypes
import uuid

def upload_file(url, file_path, field_name='files'):
    boundary = uuid.uuid4().hex
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    }
    
    data = []
    filename = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    
    data.append(f'--{boundary}'.encode())
    data.append(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode())
    data.append(f'Content-Type: {mime_type}'.encode())
    data.append(b'')
    with open(file_path, 'rb') as f:
        data.append(f.read())
    data.append(b'')
    data.append(f'--{boundary}--'.encode())
    data.append(b'')
    
    body = b'\r\n'.join(data)
    
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8'), response.status

print("Triggering processing for existing files in uploads/...")
try:
    # We upload the key again to trigger processing
    resp, status = upload_file('http://localhost:8000/process', r'c:/Users/jitin/Desktop/Desk/drive/Documents/omr/sample_key.csv', 'answer_key')
    print(f"Status: {status}")
    print(f"Response: {resp[:500]}...")
    
    if "image (1).tif" in resp:
        print("\nSUCCESS: Found 'image (1).tif' in results!")
    else:
        print("\nWARNING: 'image (1).tif' NOT found in results.")
        
except Exception as e:
    print(f"Failed: {e}")
