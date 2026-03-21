import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

base_dir = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\bin\tesseract"
fast_dir = os.path.join(base_dir, "tessdata_fast")
best_dir = os.path.join(base_dir, "tessdata_best")

os.makedirs(fast_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

models = {
    fast_dir: [
        ("ara.traineddata", "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ara.traineddata"),
        ("eng.traineddata", "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata"),
    ],
    best_dir: [
        ("ara.traineddata", "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ara.traineddata"),
        ("eng.traineddata", "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata"),
    ]
}

def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"Exists: {out_path}")
        return
    print(f"Downloading {url} to {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

for folder, files in models.items():
    for name, url in files:
        out_path = os.path.join(folder, name)
        download_file(url, out_path)

print("Done downloading models.")
