import requests, zipfile, io

url = "https://www.kaggle.com/api/v1/datasets/download/sartajbhuvaji/brain-tumor-classification-mri"
headers = {"User-Agent": "Mozilla/5.0"}

print("Downloading dataset...")
response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("dataset")
    print("✅ Dataset downloaded and extracted into 'dataset' folder")
else:
    print("❌ Failed, status:", response.status_code)
