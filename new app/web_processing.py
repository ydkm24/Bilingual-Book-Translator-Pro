"""
web_processing.py — Headless PDF processing engine for the Eel Web UI.
Wraps ocr_worker and translate_worker from workers.py without any Tkinter dependency.
"""
import os
import threading
import concurrent.futures
import subprocess
import json
import glob
import fitz  # PyMuPDF
import time

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



def worker_log(msg):
    """Simple debug logger for the background engine."""
    try:
        log_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), "PDF_Translator")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] [WebManager] {msg}\n")
    except: pass


class WebTranslatorManager:
    """Manages PDF processing lifecycle for the Eel-based Web UI."""

    def __init__(self):
        self.pdf_path = None
        self.cache_dir = None
        self.book_id = None # SPRINT 105: Store the active project ID for universal sync
        self.all_page_data = []
        self.total_pages = 0
        self.current_page = 0
        self.source_lang = "Auto-Detect"
        self.target_lang = "English"
        self.engine = "Google"
        self.is_running = False
        self.status = "ENGINE IDLE"
        self.stop_requested = False
        self.skip_requested = False
        self._worker_thread = None
        self.tesseract_path = resource_path(os.path.join("bin", "tesseract", "tesseract.exe"))

        # Fallback: if bundled tesseract not found, try system PATH
        if not os.path.exists(self.tesseract_path):
            self.tesseract_path = "tesseract"
            
        self.glossary = self._load_glossary()

    def _load_glossary(self):
        try:
            path = get_app_path("terminology.json")
            if os.path.exists(path):
                import json
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Glossary load error: {e}")
        return {}

    @property
    def active_id(self):
        """SPRINT 105: Dynamic ID Recovery.
        Returns the book_id if set, otherwise recovers it from the cache folder name.
        """
        if self.book_id:
            return self.book_id
        if self.cache_dir:
            return os.path.basename(self.cache_dir)
        return None

    # ------------------------------------------------------------------
    # PDF LOADING
    # ------------------------------------------------------------------

    def load_pdf(self, file_path, folder_name=None, cache_base=None):
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
            if not folder_name:
                import hashlib
                hasher = hashlib.md5(file_path.encode('utf-8'))
                folder_name = f"{book_name}_{hasher.hexdigest()[:8]}"
            
            from utils import get_app_path
            # Use user-specific cache_base if provided, otherwise fall back to shared
            if cache_base:
                self.cache_dir = os.path.join(cache_base, folder_name)
            else:
                self.cache_dir = get_app_path(os.path.join(".translator_cache", folder_name))
            os.makedirs(self.cache_dir, exist_ok=True)


            # Copy PDF into cache so we never lose it
            cached_pdf_path = os.path.join(self.cache_dir, "source.pdf")
            if file_path != cached_pdf_path:
                shutil.copy2(file_path, cached_pdf_path)

            self.pdf_path = cached_pdf_path
            self.total_pages = num_pages
            self.current_page = 0
            self.book_id = folder_name # SPRINT 105: Lock-on identity
            
            # SPRINT 127: Immediate Library Visibility
            try:
                info_path = os.path.join(self.cache_dir, "project_info.json")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "id": self.book_id,
                        "title": book_name,
                        "total_pages": num_pages,
                        "last_modified": time.time(),
                        "source_lang": "Auto-Detect",
                        "target_lang": "English"
                    }, f, indent=4)
                print(f"[Core] Initialized project_info.json in {self.cache_dir}")
            except Exception as e:
                print(f"[Core] Warning: Failed to initialize project_info: {e}")
            # --- SPRINT 112: GITHUB-STYLE SYNCHRONOUS PRE-LOADING ---
            # Ensure all data is in memory before returning to UI to prevent bridge hangs
            self.all_page_data = [None] * num_pages
            print(f"[Core] Synchronizing Cache RAM: Scanning for {num_pages} pages in {self.cache_dir}...")
            
            if os.path.exists(self.cache_dir):
                files = os.listdir(self.cache_dir)
                for f_name in files:
                    if f_name.startswith("page_") and f_name.endswith(".json"):
                        try:
                            # Support both padded (page_000.json) and unpadded (page_0.json)
                            idx_str = f_name.replace("page_", "").replace(".json", "")
                            idx = int(idx_str)
                            if 0 <= idx < num_pages:
                                with open(os.path.join(self.cache_dir, f_name), "r", encoding="utf-8") as f:
                                    self.all_page_data[idx] = json.load(f)
                        except Exception as e:
                            print(f"[Core] Sync Load Warning: Failed to parse {f_name}: {e}")

            print(f"[Core] Cache Synchronization Complete. Ready for UI Uplink.")
            
            # FAST SCAN: Only read project_info.json or calculate resume page from total_pages
            self.stop_requested = False
            self.is_running = False
            self.current_page = 0 # Default
            
            info_path = os.path.join(self.cache_dir, "project_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.current_page = meta.get("last_translated_page", 0)
                except: pass
            
            # Render first page image for preview (Fast)
            preview_doc = fitz.open(self.pdf_path)
            if len(preview_doc) > 0:
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
                "last_page": self.current_page,
                "cache_hash": folder_name # SPRINT 116: Explicit handle for UI
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


    def prime_detached_project(self, folder_name, title, total_pages, page_data_list, cache_base=None):
        """Initializes a project shell from cloud metadata without a source PDF."""
        # Fix: Sync with get_user_cache_path logic
        if cache_base:
            self.cache_dir = os.path.join(cache_base, folder_name)
        else:
            self.cache_dir = get_app_path(os.path.join(".translator_cache", folder_name))
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.book_id = folder_name
        self.pdf_path = None # Detached mode
        self.total_pages = total_pages
        self.current_page = 0
        self.all_page_data = [None] * total_pages
        
        # Populate with cloud data
        for p in page_data_list:
            idx = p.get("page_index")
            if idx is not None and 0 <= idx < total_pages:
                self.all_page_data[idx] = {
                    "page": idx,
                    "original": p.get("original", ""),
                    "english": p.get("english", ""),
                    "literal": p.get("literal", ""),
                    "image_url": p.get("image_url", "")
                }
        
        return {"success": True, "total_pages": total_pages, "title": title}

    def resume_detached_project(self, folder_name, cache_base=None):
        """Standardizes internal state for a PDF-less project (Mirror).
        Restores page data from local JSON cache if available.
        """
        import os
        from utils import get_app_path
        
        if cache_base:
            self.cache_dir = os.path.join(cache_base, folder_name)
        else:
            self.cache_dir = get_app_path(os.path.join(".translator_cache", folder_name))
        self.book_id = folder_name
        self.pdf_path = None
        
        if not os.path.exists(self.cache_dir):
            return {"success": False, "error": "Cache folder missing."}
            
        # Try to recover metadata
        info_path = os.path.join(self.cache_dir, "project_info.json")
        title = "Untitled Mirror"
        total_pages = 0
        if os.path.exists(info_path):
            try:
                import json
                with open(info_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                title = meta.get("title", title)
                total_pages = meta.get("total_pages", 0)
            except: pass
            
        # SPRINT 108: DISK-POWERED PAGE COUNT (Fallback)
        # Scan for existing cached pages to determine true capacity
        highest_idx = -1
        try:
            for f in os.listdir(self.cache_dir):
                if f.startswith("page_") and f.endswith(".json"):
                    try:
                        idx = int(f.replace("page_", "").replace(".json", ""))
                        highest_idx = max(highest_idx, idx)
                    except: continue
        except: pass
        
        if total_pages <= 0:
            total_pages = highest_idx + 1
        else:
            total_pages = max(total_pages, highest_idx + 1)

        self.total_pages = total_pages
        self.all_page_data = [None] * total_pages
        
        if not os.path.exists(info_path):
            return {"success": False, "error": "Project metadata missing."}
            
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            self.book_id = folder_name
            self.total_pages = meta.get("total_pages", 1)
            self.current_page = meta.get("last_translated_page", 0)
            self.all_page_data = [None] * self.total_pages
            self.pdf_path = None # Confirmed detached
            
            # SPRINT 111: Synchronize Cache RAM for Mirror Projects
            print(f"[Core] Synchronizing Mirror Cache: Scanning {self.cache_dir}...")
            files = os.listdir(self.cache_dir)
            for f_name in files:
                if f_name.startswith("page_") and f_name.endswith(".json"):
                    try:
                        idx_str = f_name.replace("page_", "").replace(".json", "")
                        idx = int(idx_str)
                        if 0 <= idx < self.total_pages:
                            with open(os.path.join(self.cache_dir, f_name), "r", encoding="utf-8") as f:
                                self.all_page_data[idx] = json.load(f)
                    except: pass
            
            return {
                "success": True, 
                "total_pages": self.total_pages, 
                "title": meta.get("title", "Untitled Mirror"),
                "last_page": self.current_page
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # PAGE IMAGE RENDERING
    # ------------------------------------------------------------------

    def get_page_image_path(self, page_idx):
        """Unified resolver for both standard and mirror image names."""
        if not self.cache_dir: return None
        
        # 1. Try Standard PDF engine name (jpg)
        img_path = os.path.join(self.cache_dir, f"img_{page_idx}.jpg")
        if os.path.exists(img_path):
            return img_path.replace("\\", "/")
            
        # 2. Try Mirror repair name (png)
        img_path_mirror = os.path.join(self.cache_dir, f"page_{page_idx}.png")
        if os.path.exists(img_path_mirror):
            return img_path_mirror.replace("\\", "/")

        # Legacy Path Support (Standard vs Padded) for PNGs
        # If the standard img_path (jpg) doesn't exist, check for PNGs
        if not os.path.exists(img_path): # Check if the JPG path exists
            # Try unpadded PNG first
            unpadded_png = os.path.join(self.cache_dir, f"page_{page_idx}.png")
            if os.path.exists(unpadded_png): return unpadded_png.replace("\\", "/")
            # Try padded PNG (Legacy v41 support)
            padded_png = os.path.join(self.cache_dir, f"page_{page_idx:03d}.png")
            if os.path.exists(padded_png): return padded_png.replace("\\", "/")

        if not self.pdf_path:
            return None

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

    def _ensure_cache_parity(self):
        """Standardizes internal state to prevent IndexErrors."""
        if not hasattr(self, 'all_page_data') or self.all_page_data is None:
            self.all_page_data = []
        if self.total_pages > 0 and len(self.all_page_data) != self.total_pages:
            # Rebuild list to exact size if mismatch detected
            current_len = len(self.all_page_data)
            if current_len < self.total_pages:
                self.all_page_data.extend([None] * (self.total_pages - current_len))
            else:
                self.all_page_data = self.all_page_data[:self.total_pages]

    # ------------------------------------------------------------------

    def get_page_data(self, page_idx):
        """Fetches OCR and Translation data for a single page (Memory-First Sync)."""
        try:
            page_idx = int(page_idx)
        except:
            return {"success": False, "error": "Invalid page index"}
            
        self._ensure_cache_parity()
        if page_idx < 0 or page_idx >= self.total_pages:
            return {"success": False, "error": "Page index out of range"}
        
        # SPRINT 112: Pre-loaded data should be present. Disk is fallback only.
        data = self.all_page_data[page_idx]
        
        if data is None:
            # Fallback for dynamic additions or edge-case sync misses
            try:
                # Try unpadded (Standard)
                path = os.path.join(self.cache_dir, f"page_{page_idx}.json")
                if not os.path.exists(path):
                    # Try padded (Legacy v41)
                    path = os.path.join(self.cache_dir, f"page_{page_idx:03d}.json")
                
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # SPRINT 111: Only cache in memory if it's NOT an empty placeholder
                        if data.get("original") or data.get("translated") or data.get("english"):
                            self.all_page_data[page_idx] = data 
                else:
                    data = {"original": "", "translated": "", "literal": ""}
            except Exception as e:
                print(f"[Core] GetPageData Error: {e}")
                return {"success": False, "error": str(e)}

        if data:
            # Cross-Engine Key Parity: Support 'source' as fallback for 'original'
            original_val = data.get("original") or data.get("source", "")
            translated_val = data.get("translated") or data.get("english", "")
            
            # SPRINT 113: ATOMIC PAYLOAD (Inline Image Base64)
            img_base64 = None
            img_path = self.get_page_image_path(page_idx)
            if img_path and os.path.exists(img_path):
                try:
                    import base64
                    from PIL import Image
                    import io
                    # SPRINT 114: NEURAL COMPRESSION (Prevent WebSocket crash)
                    with Image.open(img_path) as img:
                        # If image is very large, downscale it for the bridge
                        max_dim = 1600
                        if img.width > max_dim or img.height > max_dim:
                            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        
                        buffer = io.BytesIO()
                        # Use JPEG for best compression ratio on photographic pages
                        save_format = "JPEG" if img.mode != "RGBA" else "PNG"
                        try:
                            img.convert("RGB").save(buffer, format=save_format, quality=75)
                        except:
                            # Fallback if RGB conversion fails
                            img.save(buffer, format="PNG")
                            
                        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        mime = "image/jpeg" if save_format == "JPEG" else "image/png"
                        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        mime = "image/jpeg" if save_format == "JPEG" else "image/png"
                        img_base64 = f"data:{mime};base64,{encoded}"
                except Exception as b64_err:
                    print(f"[Core] Base64 encoding fail: {b64_err}")

            return {
                "success": True,
                "page_index": page_idx,
                "total_pages": self.total_pages,
                "image_path": img_path,
                "image_base64": img_base64, # Atomic asset delivery
                "image_url": data.get("image_url", ""), # SPRINT 111: Cloud Fallback
                "original": original_val,
                "translated": translated_val,
                "literal": data.get("literal", ""),
                "is_centered": data.get("is_centered", False),
                "is_rtl_page": data.get("is_rtl_page", False),
                "is_processing": self.is_running,
            }

    # ------------------------------------------------------------------

    def save_page_edits(self, page_idx, ocr_text, trans_text):
        """Unified save for both local PDF and Mirror projects."""
        try:
            page_idx = int(page_idx)
        except:
             return {"success": False, "error": "Invalid page index"}

        self._ensure_cache_parity()
        if page_idx < 0 or page_idx >= self.total_pages:
            return {"success": False, "error": "Page index out of range"}

        save_data = {
            "page": page_idx,
            "original": ocr_text,
            "translated": trans_text, # Standardize on "translated" for legacy parity
            "english": trans_text,    # Duplicate for mirror parity
            "literal": ""
        }
        self.all_page_data[page_idx] = save_data
        
        # Update Regular Engine Memory (if active)
        if hasattr(self, 'pages') and self.pages and 0 <= page_idx < len(self.pages):
            p = self.pages[page_idx]
            p.original = ocr_text
            p.translated = trans_text
            
        self._save_page_cache(save_data)
        return {"success": True}

    def _save_page_cache(self, page_data):
        """Synchronous disk commit for manual edits."""
        if not self.cache_dir: return
        try:
            p_idx = page_data.get("page")
            if p_idx is None: return
            path = os.path.join(self.cache_dir, f"page_{p_idx}.json")
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=4)
        except Exception as e:
            print(f"[Core] Sync failed for page {p_idx}: {e}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TRANSLATION PIPELINE
    # ------------------------------------------------------------------



    def start_translation(self, source_lang="Auto-Detect", target_lang="English", engine="Google", pages_to_process=None, ocr_tier="Standard", force=False):
        """Begins the OCR + Translation pipeline in a background thread."""
        if self.is_running:
            if force:
                print("[WebTranslator] Force restart requested. Stopping current pipeline...", flush=True)
                self.stop()
                # Busy wait for thread cleanup (max 2s)
                import time
                for _ in range(20):
                    if not self.is_running: break
                    time.sleep(0.1)
            else:
                return {"success": False, "error": "Translation already running."}

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.engine = engine
        self.ocr_tier = ocr_tier
        self.stop_requested = False
        self.skip_requested = False
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

            # SPRINT 133: Safe page_width extraction — pdf_path may be None for mirror/detached projects
            page_width = 612.0  # Default: US Letter width in points
            if self.pdf_path and os.path.exists(self.pdf_path):
                try:
                    doc = fitz.open(self.pdf_path)
                    if len(doc) > 0:
                        page_width = doc[0].rect.width
                    doc.close()
                except Exception as pw_err:
                    print(f"[WebTranslator] Warning: Could not read page width: {pw_err}", flush=True)

            # SPRINT 128: Capture session start page to force-process it if the user hit START
            # (Prevents "Skipping already processed" on the page they are looking at)
            self.session_start_page = self.current_page if pages_to_process is None else -1

            # SPRINT 105: Dynamic Queue Control
            processing_queue = list(pages_to_process) if pages_to_process is not None else []
            queue_idx = 0
            while True:
                if self.stop_requested: break
                if pages_to_process is not None:
                    if queue_idx >= len(processing_queue): break
                    page_num = processing_queue[queue_idx]
                else:
                    if self.current_page >= self.total_pages: break
                    page_num = self.current_page
                if self.stop_requested:
                    break
                
                if self.skip_requested:
                    self.skip_requested = False
                    worker_log(f"[Sync] Skip handled. Engine state already at {self.current_page}.")
                    # No UI status update for skips - keep it clean per user request
                    if pages_to_process is not None: queue_idx += 1
                    continue

                # Skip if already processed (unless we are specifically targeting it OR it's the start page of this session)
                if pages_to_process is None:
                    # FORCE process the page the user hit START on
                    if page_num != self.session_start_page:
                        existing = self.all_page_data[page_num]
                        if existing and existing.get("english"):
                            eng = existing["english"]
                            if not any(err in eng for err in ["[Translation Pending]", "[Translation Error", "[Failed]", "[OCR Error"]):
                                continue

                # Send progress update to frontend (AFTER SKIP CHECK to prevent page 1 reset)
                progress_pct = int(((page_num) / self.total_pages) * 100)
                self.status = f"OCR: PAGE {page_num + 1}/{self.total_pages}"
                try:
                    eel.update_translator_progress({
                        "page": page_num + 1,
                        "total": self.total_pages,
                        "percent": progress_pct,
                        "status": f"OCR: Scanning Page {page_num + 1} of {self.total_pages}..."
                    })()
                except:
                    pass

                # Phase 1: OCR
                # SPRINT 105: "Best" tier should never use the layout-flattening fallback (PSM 6)
                is_fallback = (pages_to_process is not None) and (self.ocr_tier != "Best")

                # SPRINT 137: Guard against None/missing pdf_path for detached/mirror projects.
                # If we have no valid source PDF, skip OCR entirely and fall through to
                # re-translate whatever OCR text is already in the cache.
                pdf_is_valid = bool(self.pdf_path and os.path.exists(self.pdf_path))

                if not pdf_is_valid:
                    # Pull existing OCR text from cache so translation can still run
                    cached = {}
                    if page_num < len(self.all_page_data):
                        cached = self.all_page_data[page_num] or {}
                    
                    cached_ocr = cached.get("original", "")
                    if cached_ocr and not any(err in cached_ocr for err in ["[OCR Error", "[Failed]", "[No Text"]):
                        print(f"[WebTranslator] No source PDF — reusing cached OCR for page {page_num}.", flush=True)
                        res = {
                            "page": page_num,
                            "original": cached_ocr,
                            "english": "[Translation Pending]",
                            "literal": cached.get("literal", ""),
                            "is_image": cached.get("is_image", False),
                            "is_rtl_page": cached.get("is_rtl_page", False),
                            "translator_src": cached.get("translator_src", translator_src),
                            "is_centered": cached.get("is_centered", False),
                            "is_cover": cached.get("is_cover", False),
                            "cover_image_path": cached.get("cover_image_path"),
                            "blocks": cached.get("blocks", []),
                        }
                    else:
                        print(f"[WebTranslator] No source PDF and no usable cache for page {page_num}. Skipping.", flush=True)
                        if pages_to_process is not None: queue_idx += 1
                        else: self.current_page = page_num + 1
                        continue
                else:
                    try:
                        is_rtl = False
                        if not is_auto and self.source_lang in LANGUAGES:
                            if self.source_lang in ["Arabic", "Hebrew", "Yiddish"]:
                                is_rtl = True
                                
                        res = ocr_worker(
                            page_num, self.pdf_path, ocr_lang_code, is_auto,
                            LANGUAGES, translator_src, page_width, is_rtl, self.tesseract_path,
                            fallback_mode=is_fallback, ocr_tier=self.ocr_tier, cache_dir=self.cache_dir,
                            cancel_check=lambda: self.stop_requested
                        )
                    except Exception as e:
                        res = {"page": page_num, "error": str(e)}

                if self.stop_requested or self.skip_requested:
                    if self.skip_requested:
                        self.skip_requested = False
                        if pages_to_process is not None: queue_idx += 1
                    if self.stop_requested: break
                    continue

                if res and "error" not in res:
                    self.all_page_data[page_num] = res
                    self._save_page_cache(res)

                    # Phase 2: Translation
                    source_text = res.get("original", "")
                    src = res.get("translator_src", translator_src)

                    if len(source_text) > 5 and "[No Text" not in source_text:
                        self.status = f"TRANS: PAGE {page_num + 1}"
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
                            glossary=self.glossary,
                            cancel_check=lambda: self.stop_requested
                        )

                        if self.stop_requested or self.skip_requested:
                            if self.skip_requested:
                                self.skip_requested = False
                                if pages_to_process is not None: queue_idx += 1
                            if self.stop_requested: break
                            continue

                        if "error" not in t_res:
                            self.all_page_data[page_num]["english"] = t_res["translated"]
                            self.all_page_data[page_num]["literal"] = t_res.get("literal", "")
                        else:
                            self.all_page_data[page_num]["english"] = f"[Translation Error: {t_res['error'][:100]}]"

                        self._save_page_cache(self.all_page_data[page_num])
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

                # Move queue forward (Absolute Indexing for Sync)
                if pages_to_process is not None: queue_idx += 1
                else: self.current_page = page_num + 1

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
                    self.status = "ENGINE IDLE"
                    status_text = "Rescan Complete!" if pages_to_process is not None else "Translation Complete!"
                    eel.update_translator_progress({
                        "page": self.total_pages,
                        "total": self.total_pages,
                        "percent": 100,
                        "status": status_text
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
            print("[DEBUG-CACHE] FAILED: No cache_dir set!", flush=True)
            return
        page_num = page_data.get("page", 0)
        cache_path = os.path.join(self.cache_dir, f"page_{page_num}.json")

        print(f"[DEBUG-CACHE] Writing Page {page_num} to {cache_path}", flush=True)
        save_data = {k: v for k, v in page_data.items() if k != "cover_image"}

        if self.pdf_path:
            save_data["original_pdf_path"] = self.pdf_path

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # SPRINT 109: HEARTBEAT FOR VERIFICATION
            heartbeat_path = os.path.join(self.cache_dir, "LAST_SAVE.txt")
            with open(heartbeat_path, "w", encoding="utf-8") as f:
                import datetime
                f.write(f"Last successful save of page {page_num} at {datetime.datetime.now()}")
                
            print(f"[DEBUG-CACHE] Write successful to {cache_path}", flush=True)
        except Exception as e:
            print(f"[DEBUG-CACHE] Failed to save page {page_num}: {e}", flush=True)

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
        """Saves manual edits for a page and flushes to local disk."""
        if page_idx < 0:
            return {"success": False, "error": "Invalid page index."}

        # SPRINT 108: ELASTIC CAPACITY
        if page_idx >= self.total_pages:
            print(f"[DEBUG-SAVE] Expanding capacity from {self.total_pages} to {page_idx + 1}", flush=True)
            extension_size = (page_idx + 1) - self.total_pages
            self.all_page_data.extend([None] * extension_size)
            self.total_pages = page_idx + 1

        if self.all_page_data[page_idx] is None:
            # Create a blank slate if they edit a completely unprocessed page
            self.all_page_data[page_idx] = {
                "page": page_idx,
                "original": ocr_text,
                "english": trans_text,
                "literal": ""
            }
        else:
            self.all_page_data[page_idx]["page"] = page_idx
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
            
            # Using strictly positional arguments for the required parameters to avoid any mismatch
            ocr_res = ocr_worker(
                0, temp_path, ocr_code, is_auto, LANGUAGES, 
                translator_src, 1000, False, self.tesseract_path,
                False, "Best", True, None # None for cache_dir in quick translate
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

    def stop_translation(self):
        """Signals the background thread to stop the entire process."""
        self.stop_requested = True
        self.is_running = False
        return {"success": True, "message": "Stop Signal Sent"}

    def set_engine_page(self, index):
        """External sync for page index (Frontend-Lead Mirroring)."""
        try:
            val = int(index)
            if 0 <= val < self.total_pages:
                self.current_page = val
                worker_log(f"[Sync] Engine shadowed to Page {val+1}")
        except: pass

    def atomic_skip_page(self, target_index):
        """Atomic Skip: Teleports the engine index and aborts in one step.
        Guarantees 1:1 synchronization with the frontend.
        """
        try: target_index = int(target_index)
        except: return {"success": False, "error": "Invalid index"}

        # SPRINT 105: Atomic Teleport
        next_idx = target_index + 1
        if next_idx < self.total_pages:
            self.current_page = next_idx
            self.skip_requested = True
            worker_log(f"[Sync] ATOMIC SKIP: Engine teleported to Page {next_idx + 1}. Aborting old work.")
            
            # Forced Abortion
            try:
                subprocess.run(["taskkill", "/F", "/IM", "tesseract.exe", "/T"], 
                               capture_output=True, check=False)
            except:
                pass
            
            return {
                "success": True, 
                "new_index": next_idx, 
                "book_id": self.book_id # Pass the active ID back to the UI
            }
        else:
            return {"success": False, "error": "Already at the end of the work."}
