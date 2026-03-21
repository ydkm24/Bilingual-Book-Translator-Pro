"""
workers.py — Standalone OCR and Translation worker functions.
These run in separate processes/threads and have no dependency on the app class.
Extracted from main.py during The Great Refactor.
"""
import os
import sys
import time
import random
import re

from utils import humanize_error, get_app_path


def ocr_worker(page_num, file_path, ocr_lang_code, is_auto, languages, translator_src, page_width, current_rtl, tesseract_path, fallback_mode=False):
    """Standalone worker for processing a single page in a separate process.
    Reads image directly from file and extracts layout blocks independently."""
    # We re-import inside the worker to ensure process-isolation safety
    import pytesseract
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb limit for large PDF pages
    import io
    from deep_translator import GoogleTranslator
    import re
    import cv2
    import numpy as np
    import fitz
    
    # CRITICAL: Path must be set PER PROCESS on Windows. Resolved via Phase A abstraction.
    # SPRINT 55: Native Path handling — TESSDATA_PREFIX must point TO the tessdata folder
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # The bundled Tesseract looks for .traineddata files directly inside TESSDATA_PREFIX,
    # it does NOT automatically append "/tessdata/". So we must point directly to it.
    tess_dir = os.path.dirname(tesseract_path)
    tess_data_dir = os.path.join(tess_dir, "tessdata")
    os.environ["TESSDATA_PREFIX"] = tess_data_dir
    
    # Simple log inside worker
    def worker_log(msg):
        try:
            log_path = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), "PDF_Translator", "debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] [Worker {os.getpid()}] Page {page_num}: {msg}\n")
        except: pass
    
    worker_log("Started processing.")
    
    # Helper functions copied/needed inside worker
    def clean_ocr_text(text):
        if not text: return ""
        # Deep clean: remove hallucinated border/line symbols but keep quotes and bullets
        text = re.sub(r'[\~\|\_\^\/\\]', ' ', text)
        # Compress repeated repetitive dots or dashes
        text = re.sub(r'\.{4,}', '.', text)
        text = re.sub(r'\-{4,}', '-', text)
            
        # Preserve double newlines for paragraph structure
        text = re.sub(r'[|\\\/_]', '', text)
        segments = text.split('\n\n')
        cleaned_segments = [re.sub(r'\s+', ' ', s).strip() for s in segments]
        return '\n\n'.join(cleaned_segments).strip()

    def detect_script(text, languages):
        text = str(text)
        # SPRINT 81: "B.S. Filter" - Remove noise and symbols to get clean script count
        clean_text = re.sub(r'[^a-zA-Z\u0600-\u06FF\u0590-\u05FF\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\s\d]', '', text)
        
        is_rtl = False
        detected_ocr = []
        detected_trans = "auto"
        
        # Ranges with character counts
        counts = {
            "heb": len(re.findall(r'[\u0590-\u05FF]', clean_text)),
            "ara": len(re.findall(r'[\u0600-\u06FF]', clean_text)),
            "ell": len(re.findall(r'[\u0370-\u03FF]', clean_text)),
            "rus": len(re.findall(r'[\u0400-\u04FF]', clean_text)),
            "chi_sim": len(re.findall(r'[\u4E00-\u9FFF]', clean_text)), # Kanji
            "jpn": len(re.findall(r'[\u3040-\u30FF]', clean_text)), # Kana
            "kor": len(re.findall(r'[\uAC00-\uD7AF]', clean_text)), # Hangul
            "latin": len(re.findall(r'[a-zA-Z\u00C0-\u017F]', clean_text))
        }
        
        total = sum(counts.values())
        if total < 5:
            return {"ocr": "eng+fra+spa+deu+ita", "trans": "auto", "rtl": False}
            
        # SPRINT 81: Hybrid Detection (Keep anything > 10% of total)
        for script, count in counts.items():
            if count > 2 or (count / total > 0.1):
                if script == "latin":
                    detected_ocr.extend(["eng", "fra", "spa", "deu", "ita"])
                else:
                    detected_ocr.append(script)
                
                if script in ["ara", "heb"]: is_rtl = True
        
        # Mapping for primary translation code
        trans_map = {"ara": "ar", "heb": "iw", "ell": "el", "rus": "ru", "chi_sim": "zh-CN", "jpn": "ja", "kor": "ko", "latin": "en"}
        
        if len(detected_ocr) > 5: # Too messy
            detected_trans = "auto"
        else:
            # Pick strongest
            strongest = max(counts, key=counts.get)
            detected_trans = trans_map.get(strongest, "auto")

        return {"ocr": "+".join(list(dict.fromkeys(detected_ocr))), "trans": detected_trans, "rtl": is_rtl}

    try:
        doc = fitz.open(file_path)
        worker_log("PDF opened.")
        page = doc.load_page(page_num)
        worker_log("Page loaded.")
        
        # Extract blocks inside the worker to prevent main-thread hangs
        blocks = list(page.get_text("blocks"))
        
        # SPRINT 81: High-Res OCR (288 DPI) for better Arabic/Asian detection
        scale_val = 8 if fallback_mode else 4
        pix = page.get_pixmap(matrix=fitz.Matrix(scale_val, scale_val))
        worker_log("Pixmap generated.")
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        worker_log("Image data ready.")
        doc.close() # Release handle immediately
        
        # ... (Logic from pass 1 and pass 2, including preprocessing and translation)
        # This is a full extraction of the page processing logic to be independent of self
        # Preprocessing
        cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cv_img is None: return {"page": page_num, "error": "Image decode failed"}
        
        # SPRINT 31: Advanced Preprocessing (Speckle Filter & Contrast for OCR Worker)
        # We use a fast version for the worker
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        worker_log("Thresholding complete.")
        
        # SPRINT 45: Vectorized Speckle Filter (Much faster for noisy PDFs)
        labels_info = cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh), connectivity=8)
        num_labels, labels, stats, _ = labels_info
        
        small_components = np.where(stats[:, cv2.CC_STAT_AREA] < 10)[0]
        mask = np.isin(labels, small_components)
        thresh[mask] = 255
        worker_log("Vectorized speckle filtering complete.")
            
        ink_density = np.sum(thresh == 0) / thresh.size
        processed_pil = Image.fromarray(thresh)
        
        # Initial state
        translated_text = ""
        literal_text = ""
        ocr_text = "" 
        page_is_rtl = current_rtl if not is_auto else False
        ocr_lang = ocr_lang_code if not is_auto else "fra+eng"
        is_real_image = False

        # SPRINT 30: Blank Page Detection
        if ink_density < 0.0005:
            is_real_image = True
            ocr_text = "[No Text]"
        else:
            if is_auto:
                worker_log("Starting script detection...")
                # SPRINT 81: Broad prescan for hybrid pages
                raw_detect = pytesseract.image_to_string(processed_pil, lang="eng+ara+heb+ell+rus+chi_sim+jpn+kor+spa+deu+ita+fra", 
                                                       config='--oem 1 --psm 3', timeout=60).strip()
                detected = detect_script(raw_detect, languages)
                ocr_lang = detected["ocr"]
                page_is_rtl = detected["rtl"]
                translator_src = detected["trans"]
                worker_log(f"Script detected: {ocr_lang}")
            
            # SPRINT 81: Dynamic PSM Logic
            if fallback_mode:
                psm_cfg = '--oem 1 --psm 6' # Try Harder mode: assume uniform block of text
            elif any(x in ocr_lang for x in ["ara", "heb", "yid"]):
                psm_cfg = '--oem 1 --psm 6' # Block mode for joined scripts
            elif any(x in ocr_lang for x in ["chi_sim", "jpn", "kor"]):
                psm_cfg = '--oem 1 --psm 3' # Column/Standard for Asian
            else:
                psm_cfg = '--oem 1 --psm 3'
            worker_log(f"Starting main OCR ({ocr_lang})...")
            try:
                raw_ocr = pytesseract.image_to_string(processed_pil, lang=ocr_lang, config=psm_cfg, timeout=60).strip()
                ocr_text = clean_ocr_text(raw_ocr)
                worker_log(f"Main OCR complete. Found {len(ocr_text)} chars.")
            except Exception as e:
                worker_log(f"Tesseract Failed: {str(e)}")
                # SPRINT 53: Increase error log length for diagnostics
                ocr_text = f"[OCR Error: {str(e)[:200]}]"
            
            # Phase 1 extraction cleanup
            if len(ocr_text) < 15: is_real_image = True
            
        # SPRINT 31: Numeral Shield for original text
        ocr_text = re.sub(r'(\d+[\-/]\d+[\-/]?[0-9]*[\-/]?[0-9]*)', u' \u2066\\1\u2069 ', ocr_text)
        
        # Phase 1: We only do Extraction now. Translation is removed from the process-level worker.

        # Detect is_image/nonsense
        is_nonsense = False
        alnum_count = 0
        if ocr_text:
            alnum_count = len(re.findall(r'[\w]', ocr_text))
            if alnum_count < len(ocr_text) * 0.3: is_nonsense = True
            
        # Fallback Logic: In Phase 1 we just ensure we have clean ocr_text.
        # Translation will be handled by Phase 2 in the main app.
        
        # Sensitivity Boost: Titles can be short (Pic 3), so we relax thresholds
        is_real_image = (len(ocr_text) < 10 and alnum_count < 5) or is_nonsense

        # Save the original image to cache for UI and Cloud Uploads
        img_path = None
        try:
            cache_dir = get_app_path(os.path.join(".translator_cache", os.path.basename(file_path).replace(".pdf", "")))
            os.makedirs(cache_dir, exist_ok=True)
            img_path = os.path.join(cache_dir, f"img_{page_num}.jpg")
            # SPRINT 98: Optimized JPEG cache (85 quality) to avoid 5GB PDFs
            img.convert("RGB").save(img_path, "JPEG", quality=85, optimize=True)
        except Exception as e:
            print(f"Failed to save image locally: {e}")

        worker_log("Finished extraction successfully.")
        return {
            "page": page_num,
            "original": ocr_text if ocr_text else "[No Text Detected]",
            "english": "[Translation Pending]",
            "literal": "[Loading...]",
            "is_image": is_real_image,
            "is_rtl_page": page_is_rtl,
            "translator_src": translator_src,
            "is_centered": (page_num == 0),
            "is_cover": (page_num == 0),
            "cover_image_path": img_path,
            "blocks": blocks # SPRINT 71: Preserve blocks for Visual Fidelity PDF
        }
    except Exception as e:
        worker_log(f"CRITICAL FAILURE: {str(e)}")
        return {"page": page_num, "error": str(e)}


def translate_worker(text, translator_src, target_lang="en", engine="Google", deepl_key="", openai_key="", ollama_model="", glossary=None, context=None, cancel_check=None):
    """Standalone worker for translating text with robust retries, throttling, and glossary enforcement.
    'context' can be a dict with 'prev_page' and 'next_page' strings for GPT-4o refinement.
    'cancel_check' is an optional callable returning True if we should abort immediately.
    """
    # SPRINT 65: Null safety for source language
    if not translator_src:
        translator_src = "auto"
    from deep_translator import GoogleTranslator
    import time
    import random
    import re
    
    # STABILITY FIX: Mandatory staggered launch to avoid "Burst" blocking
    time.sleep(random.uniform(0.1, 0.5))
    
    # Pre-processing Glossary for standard engines (Google/DeepL)
    if glossary and engine != "GPT-4o":
        for orig, target in glossary.items():
            # Use regex to replace whole words only to avoid partial matches
            pattern = re.compile(re.escape(orig), re.IGNORECASE)
            text = pattern.sub(target, text)
            
    last_err = ""
    # Try 3 times with exponential backoff
    for attempt in range(3):
        try:
            # SPRINT 31: Unicode Isolation Shield for Numbers
            text = re.sub(r'(\d[\-/]\d[\-/]?[0-9]*[\-/]?[0-9]*)', u' \u2066\\1\u2069 ', text)
            
            translated = "[Translation Error]"
            if engine == "DeepL" and deepl_key:
                import deepl
                try:
                    translator = deepl.Translator(deepl_key)
                    src = None if translator_src == 'auto' else translator_src.upper()
                    
                    # SPRINT 84: Map target language for DeepL
                    deepl_tgt = target_lang.upper()
                    if deepl_tgt == "EN": deepl_tgt = "EN-US"
                    if deepl_tgt == "ZH-CN": deepl_tgt = "ZH"
                    
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    if paragraphs:
                        res = translator.translate_text(paragraphs, target_lang=deepl_tgt, source_lang=src)
                        translated = "\n\n".join([r.text for r in res])
                    else:
                        translated = ""
                except Exception as de:
                    raise Exception(f"DeepL API Error: {de}")

            elif engine == "GPT-4o" and openai_key:
                from openai import OpenAI
                import json as pyjson
                try:
                    client = OpenAI(api_key=openai_key)
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    gloss_instr = ""
                    if glossary:
                        gloss_list = [f"'{k}' -> '{v}'" for k, v in glossary.items()]
                        gloss_instr = f"IMPORTANT: Follow this glossary: {', '.join(gloss_list)}."
                    
                    context_instr = ""
                    if context:
                        prev = context.get('prev_page', '')
                        nxt = context.get('next_page', '')
                        if prev or nxt:
                            context_instr = "\nCONTEXTUAL INFORMATION (Use for flow and consistency):\n"
                            if prev: context_instr += f"- PREVIOUS PAGE ENDING: \"{prev[-500:]}\"\n"
                            if nxt: context_instr += f"- NEXT PAGE STARTING: \"{nxt[:500]}\"\n"
                    
                    prompt = (
                        f"Translate these {len(paragraphs)} paragraphs from {translator_src} to {target_lang}. "
                        "Maintain the exact order and nuances. Use the provided glossary rules. "
                        "Return ONLY a JSON object with a key 'translations' containing an array of strings."
                    )
                    
                    response = client.chat.completions.create(
                      model="gpt-4o",
                      response_format={ "type": "json_object" },
                      messages=[
                        {"role": "system", "content": f"You are a professional literary translator. {gloss_instr}"},
                        {"role": "user", "content": f"{prompt}{context_instr}\n\nTEXT TO TRANSLATE:\n" + pyjson.dumps(paragraphs)}
                      ]
                    )
                    
                    res_raw = response.choices[0].message.content
                    res_data = pyjson.loads(res_raw)
                    translated_list = res_data.get("translations", [])
                    if len(translated_list) == len(paragraphs):
                        translated = "\n\n".join(translated_list)
                    else:
                        translated = "\n\n".join(translated_list) if translated_list else res_raw
                except Exception as ge:
                    raise Exception(f"OpenAI API Error: {ge}")
                    
            elif engine == "Ollama (Local)":
                import requests
                import json as pyjson
                try:
                    model_name = ollama_model if ollama_model else "llama3"
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    gloss_instr = ""
                    if glossary:
                        gloss_list = [f"'{k}' -> '{v}'" for k, v in glossary.items()]
                        gloss_instr = f"IMPORTANT: Follow this glossary: {', '.join(gloss_list)}."
                    
                    context_instr = ""
                    if context:
                        prev = context.get('prev_page', '')
                        nxt = context.get('next_page', '')
                        if prev or nxt:
                            context_instr = "\nCONTEXTUAL INFORMATION (Use for flow and consistency):\n"
                            if prev: context_instr += f"- PREVIOUS PAGE ENDING: \"{prev[-500:]}\"\n"
                            if nxt: context_instr += f"- NEXT PAGE STARTING: \"{nxt[:500]}\"\n"
                    
                    prompt = (
                        f"Translate exactly these {len(paragraphs)} paragraphs from {translator_src} to {target_lang}. "
                        "DO NOT SUMMARIZE. DO NOT OMIT ANYTHING. Maintain the EXACT order, length, and nuances. "
                        f"Use the provided glossary rules. Return ONLY a JSON object with a key 'translations' containing exactly {len(paragraphs)} translated strings."
                    )
                    
                    payload = {
                        "model": model_name,
                        "format": "json",
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": f"You are a highly precise literal translator. You never summarize. You translate every single sentence exactly as written. {gloss_instr}"},
                            {"role": "user", "content": f"Translate exactly 2 paragraphs from hebrew to english. DO NOT SUMMARIZE. Return ONLY a JSON object with a key 'translations' containing exactly 2 translated strings.\n\nTEXT TO TRANSLATE:\n[\"שלום\", \"להתראות\"]"},
                            {"role": "assistant", "content": "{\"translations\": [\"Hello\", \"Goodbye\"]}"},
                            {"role": "user", "content": f"{prompt}{context_instr}\n\nTEXT TO TRANSLATE:\n" + pyjson.dumps(paragraphs)}
                        ]
                    }
                    
                    # Local inference can be extremely slow on CPU or with heavy models, use a massive timeout.
                    response = requests.post(
                        "http://127.0.0.1:11434/api/chat", 
                        json=payload, 
                        timeout=1200, 
                        proxies={"http": None, "https": None}
                    )
                    
                    if response.status_code == 404:
                        raise Exception(f"Model '{model_name}' not found in Ollama or endpoint missing. Check spelling!")
                        
                    response.raise_for_status()
                    
                    res_data = response.json()
                    res_raw = res_data.get("message", {}).get("content", "{}")
                    res_json = pyjson.loads(res_raw)
                    translated_list = res_json.get("translations", [])
                    
                    if len(translated_list) == len(paragraphs):
                        translated = "\n\n".join(translated_list)
                    else:
                        translated = "\n\n".join(translated_list) if translated_list else res_raw
                except Exception as oe:
                    raise Exception(f"Ollama Local Error: {oe}. Is Ollama running and '{model_name}' installed?")
                    
            else: # Default Google
                if translator_src == "heb": translator_src = "iw"
                
                # SPRINT 82: Restore simple GoogleTranslator usage
                translator = GoogleTranslator(source=translator_src, target=target_lang)
                
                # Split by double newlines to preserve paragraph structure
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                translated_paragraphs = []
                
                for i, p in enumerate(paragraphs):
                    if cancel_check and cancel_check():
                        return {"translated": "[Translation Stopped]", "literal": ""}
                    
                    if i > 0:
                        time.sleep(random.uniform(0.5, 1.2)) # Basic jitter
                        
                    res = translator.translate(p)
                    if not res: raise Exception("Google returned empty result.")
                    translated_paragraphs.append(res)
                
                translated = "\n\n".join(translated_paragraphs)
            
            if not translated or not translated.strip():
                raise Exception("Translation engine returned an empty result.")

            words = text.split()
            literal = " ".join(words[:50]) + ("..." if len(words) > 50 else "")
            
            return {"translated": translated, "literal": literal}
        except Exception as e:
            last_err = humanize_error(e)
            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(wait_time)
            
    return {"error": f"Failed after 3 retries: {last_err}"}
