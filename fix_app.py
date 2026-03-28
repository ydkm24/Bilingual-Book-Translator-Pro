import os, json

app_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """        if is_local:
            # SPRINT 132: Clone from local cache
            project_dir = os.path.join(base_path, book_id)
            if not os.path.exists(project_dir):
                return {"success": False, "error": "Local project not found."}
                
            for f in os.listdir(project_dir):
                if f.startswith("page_") and f.endswith(".json"):
                    with open(os.path.join(project_dir, f), "r", encoding="utf-8") as jf:
                        page_data = json.load(jf)
                        # We need original image (base64 or path), original text, english text
                        img_path = page_data.get("image_path", "")
                        img_base64 = ""
                        if img_path and os.path.exists(img_path):
                            import base64
                            try:
                                with open(img_path, "rb") as image_file:
                                    img_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                            except: pass
                            
                        pages.append({
                            "page_index": page_data.get("page_index", 0),
                            "original": page_data.get("original", ""),
                            "english": page_data.get("translated", ""),
                            "image_url": img_base64 or img_path
                        })"""

replacement = """        if is_local:
            # SPRINT 132: Clone from local cache
            project_dir = os.path.join(base_path, book_id)
            if not os.path.exists(project_dir):
                return {"success": False, "error": "Local project not found."}
                
            import glob
            for f_path in glob.glob(os.path.join(project_dir, "page_*.json")):
                with open(f_path, "r", encoding="utf-8") as jf:
                    page_data = json.load(jf)
                    
                    p_idx = page_data.get("page", 0)
                    
                    # Dynamically resolve images
                    img_path = os.path.join(project_dir, f"img_{p_idx}.jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(project_dir, f"page_{p_idx}.png")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(project_dir, f"page_{p_idx:03d}.png")
                        if not os.path.exists(img_path):
                            img_path = ""
                            
                    img_base64 = ""
                    if img_path and os.path.exists(img_path):
                        import base64
                        try:
                            with open(img_path, "rb") as image_file:
                                mime = "image/png" if img_path.endswith(".png") else "image/jpeg"
                                img_base64 = f"data:{mime};base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                        except: pass
                        
                    pages.append({
                        "page_index": p_idx,
                        "original": page_data.get("original", ""),
                        "english": page_data.get("translated", "") or page_data.get("english", ""),
                        "image_url": img_base64 or img_path
                    })"""

if target in content:
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(content.replace(target, replacement))
    print("SUCCESS: REPLACED APP.PY TARGET CONTENT")
else:
    print("FAILED: TARGET NOT FOUND IN APP.PY CONTENT")
