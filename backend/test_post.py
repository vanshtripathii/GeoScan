import requests, base64
from PIL import Image
import io

# Create two tiny images
img1 = Image.new('RGB', (10,10), (255,0,0))
img2 = Image.new('RGB', (10,10), (0,0,255))

buf1 = io.BytesIO()
img1.save(buf1, format='PNG')
b1 = base64.b64encode(buf1.getvalue()).decode()

buf2 = io.BytesIO()
img2.save(buf2, format='PNG')
b2 = base64.b64encode(buf2.getvalue()).decode()

payload = {
    'image1': 'data:image/png;base64,' + b1,
    'image2': 'data:image/png;base64,' + b2
}

resp = requests.post('http://localhost:5000/detect-changes', json=payload, timeout=10)
print('Status:', resp.status_code)
print('Response:', resp.text)
