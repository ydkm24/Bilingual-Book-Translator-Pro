"""
web_processing.py — Headless PDF processing engine for the Eel Web UI.
Wraps ocr_worker and translate_worker from workers.py without any Tkinter dependency.
"""
import os
import threading
import concurrent.futures
import json
import glob
import fitz  # PyMuPDF

import eel
from utils import resource_path, get_app_path
from workers import ocr_worker, translate_worker


# Full language map (mirrors main.py)
LANGUAGES = {
    "Auto-Detect": {"ocr": "eng", "trans": "auto"},
    "English": {"ocr": "eng", "trans": "en"},
    "Spanish": {"ocr": "spa", "trans": "es"},
    "French": {"ocr": "fra", "trans": "fr"},
    "German": {"ocr": "deu", "trans": "de"},
    "Italian": {"ocr": "ita", "trans": "it"},
    "Portuguese": {"ocr": "por", "trans": "pt"},
    "Russian": {"ocr": "rus", "trans": "ru"},
    "Chinese": {"ocr": "chi_sim", "trans": "zh-CN"},
    "Japanese": {"ocr": "jpn", "trans": "ja"},
    "Korean": {"ocr": "kor", "trans": "ko"},
    "Arabic": {"ocr": "ara", "trans": "ar"},
    "Hebrew": {"ocr": "heb", "trans": "iw"},
    "Greek": {"ocr": "ell", "trans": "el"},
    "Turkish": {"ocr": "tur", "trans": "tr"},
    "Dutch": {"ocr": "nld", "trans": "nl"},
    "Polish": {"ocr": "pol", "trans": "pl"},
    "Latin": {"ocr": "lat", "trans": "la"},
    "Yiddish": {"ocr": "yid", "trans": "yi"},
}


class WebTranslatorManager:
    """Manages PDF processing lifecycle for the Eel-based Web UI."""

    def __init__(self):
        self.pdf_path = None
        self.cache_dir = None
        self.all_page_data = []
        self.total_pages = 0
        self.current_page = 0
        self.source_lang = "Auto-Detect"
        self.target_lang = "English"
        self.engine = "Google"
        self.is_running = False
        self.stop_requested = False
        self._worker_thread = None
        self.tesseract_path = resource_path(os.path.join("bin", "tesseract", "tesseract.exe"))

        # Fallback: if bundled tesseract not found, try system PATH
        if not os.path.exists(self.tesseract_path):
            self.tesseract_path = "tesseract"

    # ------------------------------------------------------------------
    # PDF LOADING
    # ------------------------------------------------------------------

    def load_pdf(self, file_path):
        """Opens a PDF, extracts page count, renders first page image, returns metadata."""
        if not file_path or not os.path.exists(file_path):
            return {"success": False, "error": "File not found."}

        try:
            import fitz
            import shutil
            import json
            import time

            doc = fitz.open(file_path)
            num_pages = len(doc)
            doc.close()

            # Setup cache
            book_name = os.path.basename(file_path).replace(".pdf", "")
            
            # Create a unique but consistent cache dir name
            import hashlib
            hasher = hashlib.md5(file_path.encode('utf-8'))
            folder_name = f"{book_name}_{hasher.hexdigest()[:8]}"
            
            from utils import get_app_path
            self.cache_dir = get_app_path(os.path.join(".translator_cache", folder_name))
            os.makedirs(self.cache_dir, exist_ok=True)

            # Copy PDF into cache so we never lose it
            cached_pdf_path = os.path.join(self.cache_dir, "source.pdf")
            if file_path != cached_pdf_path:
                shutil.copy2(file_path, cached_pdf_path)

            self.pdf_path = cached_pdf_path
            self.total_pages = num_pages
            self.current_page = 0
            self.all_page_data = [None] * num_pages

            # Generate/Update project_info.json
            info_path = os.path.join(self.cache_dir, "project_info.json")
            if not os.path.exists(info_path):
                metadata = {
                    "book_name": book_name,
                    "total_pages": num_pages,
                    "last_translated_page": 0,
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "last_accessed": time.time(),
                    "cache_folder": folder_name
                }
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
            else:
                # Update timestamp
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata["last_accessed"] = time.time()
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4)
                except:
                    pass
            self.stop_requested = False
            self.is_running = False

            # Render first page image for preview
            preview_doc = fitz.open(self.pdf_path)
            page = preview_doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            img_path = os.path.join(self.cache_dir, "img_0.jpg")
            pix.save(img_path)
            preview_doc.close()

            return {
                "success": True,
                "total_pages": num_pages,
                "title": book_name,
                "first_image": img_path.replace("\\", "/"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # PAGE IMAGE RENDERING
    # ------------------------------------------------------------------

    def get_page_image_path(self, page_idx):
        """Returns the cached image path for a page, rendering it if needed."""
        if not self.pdf_path:
            return None

        img_path = os.path.join(self.cache_dir, f"img_{page_idx}.jpg")
        if os.path.exists(img_path):
            return img_path.replace("\\", "/")

        # Render on demand
        try:
            doc = fitz.open(self.pdf_path)
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            pix.save(img_path)
            doc.close()
            return img_path.replace("\\", "/")
        except:
            return None

    # ------------------------------------------------------------------
    # PAGE DATA ACCESS
    # ------------------------------------------------------------------

    def get_page_data(self, page_idx):
        """Returns cached page data (OCR text, translation) for a given page index."""
        if page_idx < 0 or page_idx >= self.total_pages:
            return {"success": False, "error": "Page out of range."}

        self.current_page = page_idx
        img_path = self.get_page_image_path(page_idx)

        data = self.all_page_data[page_idx] if page_idx < len(self.all_page_data) else None

        return {
            "success": True,
            "page_index": page_idx,
            "total_pages": self.total_pages,
            "image_path": img_path,
            "original": data.get("original", "") if data else "",
            "translated": data.get("english", "") if data else "",
            "literal": data.get("literal", "") if data else "",
            "is_centered": data.get("is_centered", False) if data else False,
            "is_rtl_page": data.get("is_rtl_page", False) if data else False,
            "is_processing": self.is_running,
        }

    # ------------------------------------------------------------------
    # TRANSLATION PIPELINE
    # ------------------------------------------------------------------

    def rebalance_scripts(self, text):
        """Fixes common French/Greek script confusion and bracket mangling."""
        if not text: return ""
        import re
        text = re.sub(r'\bἰ[οo]\b', 'le', text)
        text = re.sub(r'\bἰ[δθ]\b', 'le', text)
        text = re.sub(r' ([αα]ο|οα) ', ' le ', text)
        text = re.sub(r'\({2,}', '(', text)
        text = re.sub(r'\){2,}', ')', text)
        text = re.sub(r'\(\(\. ', '(f. ', text)
        text = re.sub(r'6\"-7', '6-7', text)
        text = re.sub(r'\bἰο\b', 'le', text)
        text = re.sub(r'ἰο tome', 'le tome', text)
        return text

    def start_translation(self, source_lang="Auto-Detect", target_lang="English", engine="Google", pages_to_process=None):
        """Begins the OCR + Translation pipeline in a background thread."""
        if self.is_running:
            return {"success": False, "error": "Translation already running."}
        if not self.pdf_path:
            return {"success": False, "error": "No PDF loaded."}

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.engine = engine
        self.stop_requested = False
        self.is_running = True

        self._worker_thread = threading.Thread(target=self._run_pipeline, args=(pages_to_process,), daemon=True)
        self._worker_thread.start()

        return {"success": True, "message": "Translation started."}

    def _run_pipeline(self, pages_to_process=None):
        """The main OCR + Translation loop, runs in a background thread."""
        try:
            lang_info = LANGUAGES.get(self.source_lang, {"ocr": "eng", "trans": "auto"})
            target_info = LANGUAGES.get(self.target_lang, {"trans": "en"})
            ocr_lang_code = lang_info["ocr"]
            translator_src = lang_info["trans"]
            target_trans = target_info["trans"]
            is_auto = (self.source_lang == "Auto-Detect")

            doc = fitz.open(self.pdf_path)
            page_width = doc[0].rect.width
            doc.close()

            loop_iterable = pages_to_process if pages_to_process is not None else range(self.total_pages)

            for page_num in loop_iterable:
                if self.stop_requested:
                    break

                # Send progress update to frontend
                progress_pct = int(((page_num) / self.total_pages) * 100)
                try:
                    eel.update_translator_progress({
                        "page": page_num + 1,
                        "total": self.total_pages,
                        "percent": progress_pct,
                        "status": f"OCR: Scanning Page {page_num + 1} of {self.total_pages}..."
                    })()
                except:
                    pass

                # Skip if already processed (unless we are specifically targeting it)
                if pages_to_process is None:
                    existing = self.all_page_data[page_num]
                    if existing and existing.get("english"):
                        eng = existing["english"]
                        # Do NOT skip if the page has a pending or error marker
                        if not any(err in eng for err in ["[Translation Pending]", "[Translation Error", "[Failed]", "[OCR Error"]):
                            continue

                # Phase 1: OCR
                is_fallback = (pages_to_process is not None)
                try:
                    res = ocr_worker(
                        page_num, self.pdf_path, ocr_lang_code, is_auto,
                        LANGUAGES, translator_src, page_width, False, self.tesseract_path,
                        fallback_mode=is_fallback
                    )
                except Exception as e:
                    res = {"page": page_num, "error": str(e)}

                if self.stop_requested:
                    break

                if res and "error" not in res:
                    # Clean up common OCR artifacts before translation
                    if res.get("original"):
                        res["original"] = self.rebalance_scripts(res["original"])

                    self.all_page_data[page_num] = res
                    self._save_page_cache(res)

                    # Phase 2: Translation
                    source_text = res.get("original", "")
                    src = res.get("translator_src", translator_src)

                    if len(source_text) > 5 and "[No Text" not in source_text:
                        try:
                            eel.update_translator_progress({
                                "page": page_num + 1,
                                "total": self.total_pages,
                                "percent": progress_pct,
                                "status": f"Translating Page {page_num + 1}..."
                            })()
                        except:
                            pass

                        t_res = translate_worker(
                            text=source_text,
                            translator_src=src,
                            target_lang=target_trans,
                            engine=self.engine,
                        )

                        if self.stop_requested:
                            break

                        if "error" not in t_res:
                            self.all_page_data[page_num]["english"] = t_res["translated"]
                            self.all_page_data[page_num]["literal"] = t_res.get("literal", "")
                        else:
                            self.all_page_data[page_num]["english"] = f"[Translation Error: {t_res['error'][:100]}]"

                        self._save_page_cache(self.all_page_data[page_num])
                        
                        # Stealth throttle to avoid hitting Google Translate API limits
                        import time
                        import random
                        time.sleep(3.0 + random.uniform(0.1, 0.5))
                    else:
                        # Image-only or blank page
                        self.all_page_data[page_num]["english"] = source_text
                        self.all_page_data[page_num]["literal"] = ""
                        self._save_page_cache(self.all_page_data[page_num])
                else:
                    err_msg = res.get("error", "Unknown") if res else "Worker returned None"
                    self.all_page_data[page_num] = {
                        "page": page_num,
                        "original": f"[OCR Error: {err_msg[:100]}]",
                        "english": "[Failed]",
                        "literal": "",
                        "is_image": False,
                    }

                # Notify frontend of page completion
                try:
                    eel.update_translator_page_done({
                        "page_index": page_num,
                        "data": {
                            "original": self.all_page_data[page_num].get("original", ""),
                            "translated": self.all_page_data[page_num].get("english", ""),
                            "literal": self.all_page_data[page_num].get("literal", ""),
                            "is_centered": self.all_page_data[page_num].get("is_centered", False),
                            "is_rtl_page": self.all_page_data[page_num].get("is_rtl_page", False)
                        }
                    })()
                except:
                    pass

            # Final progress calculation
            try:
                if self.stop_requested:
                    # Retain current position if paused
                    stop_pct = int((page_num / self.total_pages) * 100) if self.total_pages else 0
                    eel.update_translator_progress({
                        "page": page_num,
                        "total": self.total_pages,
                        "percent": stop_pct,
                        "status": f"Translation Paused at page {page_num} of {self.total_pages}."
                    })()
                else:
                    # Completed successfully
                    eel.update_translator_progress({
                        "page": self.total_pages,
                        "total": self.total_pages,
                        "percent": 100,
                        "status": "Translation Complete!"
                    })()
            except:
                pass

        except Exception as e:
            print(f"[WebTranslator] Pipeline error: {e}", flush=True)
            try:
                eel.update_translator_progress({
                    "page": 0, "total": 0, "percent": 0,
                    "status": f"Pipeline Error: {str(e)[:100]}"
                })()
            except:
                pass
        finally:
            self.is_running = False

    def _save_page_cache(self, page_data):
        """Saves a page's data to the local JSON cache."""
        if not self.cache_dir:
            return
        page_num = page_data.get("page", 0)
        cache_path = os.path.join(self.cache_dir, f"page_{page_num}.json")

        save_data = {k: v for k, v in page_data.items() if k != "cover_image"}

        if self.pdf_path:
            save_data["original_pdf_path"] = self.pdf_path

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Failed to save page {page_num}: {e}")

    # ------------------------------------------------------------------
    # CONTROLS
    # ------------------------------------------------------------------

    def stop(self):
        """Stops the running pipeline gracefully."""
        self.stop_requested = True

    def clear(self):
        """Fully resets the manager state."""
        self.stop_requested = True
        self.is_running = False
        self.pdf_path = None
        self.cache_dir = None
        self.all_page_data = []
        self.total_pages = 0
        self.current_page = 0
        
    def save_page_edits(self, page_idx, ocr_text, trans_text):
        """Allows user to manually edit the OCR or Translated text of a page."""
        if page_idx < 0 or page_idx >= self.total_pages:
            return {"success": False, "error": "Invalid page index."}
            
        if self.all_page_data[page_idx] is None:
            # Create a blank slate if they edit a completely unprocessed page
            self.all_page_data[page_idx] = {
                "page": page_idx,
                "original": ocr_text,
                "english": trans_text,
                "literal": ""
            }
        else:
            self.all_page_data[page_idx]["original"] = ocr_text
            self.all_page_data[page_idx]["english"] = trans_text
            
        self._save_page_cache(self.all_page_data[page_idx])
        return {"success": True}

    # ------------------------------------------------------------------
    # QUICK TRANSLATE 
    # ------------------------------------------------------------------

    def quick_translate_text(self, text, source_lang, target_lang, engine):
        """Quickly translates a snippet of text off-thread to avoid blocking."""
        if not text.strip():
            return {"success": False, "error": "No text provided."}
            
        lang_info = LANGUAGES.get(source_lang, {"trans": "auto"})
        target_info = LANGUAGES.get(target_lang, {"trans": "en"})
        src = lang_info["trans"]
        tgt = target_info["trans"]
        
        try:
            res = translate_worker(text, src, tgt, engine)
            if "error" in res:
                return {"success": False, "error": res["error"]}
            return {"success": True, "translated": res["translated"], "literal": res.get("literal", "")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def quick_translate_image(self, base64_data, source_lang, target_lang, engine):
        """Saves base64 image, runs OCR on it, then translates."""
        import base64
        import tempfile
        import time
        from PIL import Image
        import io

        if not base64_data:
            return {"success": False, "error": "No image data."}

        try:
            # Strip data url prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            
            img_bytes = base64.b64decode(base64_data)
            
            # Save to temporary file for Tesseract
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"qt_{int(time.time())}.png")
            
            image = Image.open(io.BytesIO(img_bytes))
            image.save(temp_path, "PNG")
            
            # Run OCR
            lang_info = LANGUAGES.get(source_lang, {"ocr": "eng", "trans": "auto"})
            target_info = LANGUAGES.get(target_lang, {"trans": "en"})
            
            ocr_code = lang_info["ocr"]
            is_auto = (source_lang == "Auto-Detect")
            translator_src = lang_info["trans"]
            tgt_trans = target_info["trans"]
            
            # Just pass arbitrary page num and width
            ocr_res = ocr_worker(
                0, temp_path, ocr_code, is_auto, LANGUAGES, 
                translator_src, page_width=1000, 
                is_quick_mode=True, tesseract_path=self.tesseract_path
            )
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
                
            if "error" in ocr_res:
                return {"success": False, "error": ocr_res["error"]}
                
            ocr_text = ocr_res.get("original", "").strip()
            if not ocr_text:
                return {"success": False, "error": "No text detected in image."}
                
            # Translate
            t_res = translate_worker(ocr_text, ocr_res.get("translator_src", translator_src), tgt_trans, engine)
            
            if "error" in t_res:
                return {"success": False, "error": t_res["error"]}
                
            return {
                "success": True, 
                "original": ocr_text,
                "translated": t_res["translated"],
                "literal": t_res.get("literal", ""),
                "detected_lang": ocr_res.get("translator_src", translator_src)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search_text(self, query):
        """Searches all cached pages for the given text query, returns matching page indices."""
        if not query or not self.all_page_data:
            return {"success": False, "results": []}

        query_lower = query.lower()
        matches = []

        for i, data in enumerate(self.all_page_data):
            if data is None:
                continue
            original = (data.get("original", "") or "").lower()
            translated = (data.get("english", "") or "").lower()
            if query_lower in original or query_lower in translated:
                # Return a snippet for context
                text = data.get("original", "") + " " + data.get("english", "")
                idx = text.lower().find(query_lower)
                snippet_start = max(0, idx - 40)
                snippet_end = min(len(text), idx + len(query) + 40)
                snippet = "..." + text[snippet_start:snippet_end] + "..."

                matches.append({"page": i, "snippet": snippet})

        return {"success": True, "results": matches}
