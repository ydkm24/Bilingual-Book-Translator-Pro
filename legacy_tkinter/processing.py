"""
processing.py - PDF processing pipeline, OCR loop, page rendering, and navigation.
Extracted from main.py during The Great Refactor as a mixin class.
"""
import os
import re
import json
import time
import random
import threading
import concurrent.futures
import multiprocessing
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageFilter, ImageTk
import io
import fitz
import cv2
import numpy as np
import pytesseract
from functools import partial
from typing import Any, cast

from utils import get_app_path, resource_path, humanize_error, PAPER_BG, TEXT_COLOR, TEXT_DIM, ACCENT_COLOR, PAPER_SHEET_BG, CARD_BG, FRAME_BG, BTN_PRIMARY, BTN_PRIMARY_HOVER, BTN_SECONDARY, BTN_SECONDARY_HOVER, BTN_DANGER, BTN_SUCCESS, INPUT_BG
from workers import ocr_worker, translate_worker
from deep_translator import GoogleTranslator


class ProcessingMixin:
    """Mixin class providing PDF processing, OCR, translation pipeline, rendering, and navigation."""

    def close_book(self):
        """SPRINT 38: Handle explicit book closure and cleanup."""
        self.save_edits_to_cache()
        self.stop_requested = True
        
        # SPRINT 43: Release lock if it's a cloud book (Backgrounded in v1.4.1 to avoid UI hang)
        self.heartbeat_active = False # Stop pinging
        if getattr(self, "current_book_id", None) and self.supabase and self.current_user:
            def _release_lock(b_id, user_id):
                try:
                    # Only release if WE hold the lock
                    res = self.supabase.table("books").select("active_editor_id").eq("id", b_id).execute()
                    if res.data and res.data[0].get('active_editor_id') == user_id:
                        self.supabase.table("books").update({
                            "active_editor_id": None,
                            "active_editor_time": None
                        }).eq("id", b_id).execute()
                except Exception as e:
                    print(f"Failed to release book lock: {e}")
            
            threading.Thread(target=_release_lock, args=(self.current_book_id, self.current_user['id']), daemon=True).start()
                
        self.all_page_data = []
        self.image_cache = {}
        self.current_page_idx = 0
        self.total_pages = 0
        self.current_pdf_path = None
        self.current_book_id  = None
        self.cache_dir = None
        self.last_cache_dir = getattr(self, "cache_dir", "") or self.last_cache_dir
        self.last_original_path = getattr(self, "current_pdf_path", "") or self.last_original_path
        self.last_book_id = getattr(self, "current_book_id", "") or self.last_book_id
        
        # Reset UI (SPRINT 79: restored missing method)
        try:
            self.reset_ui(hard_reset=True)
        except Exception as e:
            print(f"UI Reset failed: {e}")
            
        self.status_label.configure(text="Book closed.")
        self.menu_bar.entryconfigure("Export", state="disabled")
        self.file_menu.entryconfigure("↺ Resume Last Session", state="disabled") # Disabled until next open
        try:
            self.save_settings()
        except: pass

    def check_existing_session(self):
        """Checks if there's an existing cache on disk and shows Resume button if so."""
        import glob
        base_cache = get_app_path(".translator_cache")
        
        # Disable resume initially
        try: self.file_menu.entryconfigure("↺ Resume Last Session", state="disabled")
        except: pass
        
        if os.path.exists(base_cache):
            # SPRINT 46: Prioritize Last Used Session
            if getattr(self, 'last_cache_dir', None) and os.path.exists(self.last_cache_dir):
                files = glob.glob(os.path.join(self.last_cache_dir, "page_*.json"))
                if files:
                    self.cache_dir = self.last_cache_dir
                    self.file_menu.entryconfigure("↺ Resume Last Session", state="normal")
                    self.status_label.configure(text=f"Ready to resume latest ({len(files)} pages).")
                    return

            # Fallback to search if last_cache_dir is missing/invalid
            for sub in os.listdir(base_cache):
                sub_path = os.path.join(base_cache, sub)
                if os.path.isdir(sub_path):
                    files = glob.glob(os.path.join(sub_path, "page_*.json"))
                    if files:
                        self.cache_dir = sub_path  # Point to the correct book subfolder
                        self.file_menu.entryconfigure("↺ Resume Last Session", state="normal")
                        self.status_label.configure(text=f"Found existing session ({len(files)} pages).")
                        return

        # SPRINT 47: Global Library Fallback
        if getattr(self, "last_book_id", None) and self.supabase:
            def _check_cloud():
                try:
                    res = self.supabase.table("books").select("*").eq("id", self.last_book_id).execute()
                    if res.data:
                        book = res.data[0]
                        self.schedule_ui_update(lambda: self.file_menu.entryconfigure("↺ Resume Last Session", state="normal"))
                        self.schedule_ui_update(lambda: self.status_label.configure(text=f"Ready to resume '{book['title']}' from cloud."))
                except: pass
            threading.Thread(target=_check_cloud, daemon=True).start()

    def resume_session(self):
        """Loads all page JSON files from cache directly into memory."""
        import glob
        import json
        import re
        
        # SPRINT 47: Handle cloud-only resumption
        if not getattr(self, "cache_dir", None) and getattr(self, "last_book_id", None) and self.supabase:
            try:
                res = self.supabase.table("books").select("*").eq("id", self.last_book_id).execute()
                if res.data:
                    self.load_book_from_cloud(res.data[0])
                    return
            except: pass

        if not getattr(self, "cache_dir", None):
             messagebox.showwarning("Resume Failed", "No local or cloud session found to resume.")
             return
             
        files = glob.glob(os.path.join(self.cache_dir, "page_*.json"))
        if not files: return
        
        self.status_label.configure(text="Loading session from disk...")
        self.progress_bar.set(0)
        self.all_page_data = []
        self.image_cache = {}
        
        def load_cache():
            try:
                # Sort files by the page number in their filename
                def extract_num(f):
                    match = re.search(r'page_(\d+)\.json', f)
                    return int(match.group(1)) if match else 0
                    
                sorted_files = sorted(files, key=extract_num)
                
                total_f = len(sorted_files)
                for i, f in enumerate(sorted_files):
                    with open(f, 'r', encoding='utf-8') as jfile:
                        data = json.load(jfile)
                        page_idx = data.get('page', 0)
                        
                        # Ensure the list is large enough to insert at this index
                        while len(self.all_page_data) <= page_idx:
                            self.all_page_data.append(None)
                            
                        self.all_page_data[page_idx] = data
                        
                        # Update progress bar
                        if total_f > 0:
                            self.schedule_ui_update(lambda v=(i+1)/total_f: self.progress_bar.set(v))
                            
                self.total_pages = len(self.all_page_data)
                
                # Ready to view
                self.schedule_ui_update(lambda: self.status_label.configure(text="Session Restored!"))
                self.schedule_ui_update(lambda: self.menu_bar.entryconfigure("Export", state="normal"))
                
                # Set the app to a paused state so the user can click "Resume"
                self.is_paused = True
                self.stop_requested = True
                self.schedule_ui_update(lambda: self.pause_btn.configure(state="normal", text="Resume", fg_color="#2E7D32", hover_color="#1B5E20", command=self.toggle_pause))
                
                # Automatically pull the original PDF path from the first page's metadata
                if self.all_page_data and 'original_pdf_path' in self.all_page_data[0]:
                    cache_path = self.all_page_data[0]['original_pdf_path']
                    found_path = None
                    
                    if os.path.exists(cache_path):
                        found_path = cache_path
                    else:
                        # SPRINT 46: Smart PDF Recovery
                        self.schedule_ui_update(lambda: self.status_label.configure(text="Searching for missing PDF..."))
                        filename = os.path.basename(cache_path)
                        search_paths = [
                            get_app_path("Downloads"),
                            os.path.join(os.path.expanduser("~"), "Downloads"),
                            os.path.join(os.path.expanduser("~"), "Documents"),
                            os.getcwd()
                        ]
                        for sp in search_paths:
                            trial = os.path.join(sp, filename)
                            if os.path.exists(trial):
                                found_path = trial
                                break
                                
                    if found_path:
                        self.current_pdf_path = found_path
                        # SPRINT 100: Establishing the ULTIMATE source of truth - the PDF itself
                        try:
                            temp_doc = fitz.open(found_path)
                            actual_count = len(temp_doc)
                            temp_doc.close()
                            # Pad or truncate to match physical document
                            if len(self.all_page_data) < actual_count:
                                self.all_page_data.extend([{"page": i} for i in range(len(self.all_page_data), actual_count)])
                            elif len(self.all_page_data) > actual_count:
                                self.all_page_data = self.all_page_data[:actual_count]
                            self.total_pages = actual_count
                            print(f"[Resume] Re-acquiring PDF truth: {actual_count} pages.")
                        except: pass
                    else:
                        # ULTIMATE FALLBACK: Manual Selection
                        from tkinter import filedialog
                        def _manual_find():
                            # We must do this on main thread or it might hang/hidden
                            f_name = os.path.basename(cache_path)
                            self.schedule_ui_update(lambda: messagebox.showinfo("Locate PDF", f"Session found, but original file is missing:\n{f_name}\n\nPlease select the PDF to resume."))
                            selected = filedialog.askopenfilename(
                                title=f"Locate {f_name}",
                                filetypes=[("PDF Files", "*.pdf")]
                            )
                            if selected:
                                self.current_pdf_path = selected
                                # SPRINT 100: Same truth enforcement for manual selection
                                try:
                                    temp_doc = fitz.open(selected)
                                    actual_count = len(temp_doc)
                                    temp_doc.close()
                                    if len(self.all_page_data) < actual_count:
                                        self.all_page_data.extend([{"page": i} for i in range(len(self.all_page_data), actual_count)])
                                    elif len(self.all_page_data) > actual_count:
                                        self.all_page_data = self.all_page_data[:actual_count]
                                    self.total_pages = actual_count
                                except: pass
                                self.status_label.configure(text=f"Session Restored ({actual_count} pages)")
                            else:
                                self.schedule_ui_update(lambda: messagebox.showwarning("File Missing", "Original PDF missing. Rescan/OCR features will be disabled."))
                        
                        self.after(500, _manual_find)
                
                if self.all_page_data:
                    # SPRINT 63: Smart Resume Index - find where we left off
                    smart_idx = 0
                    for i, pd in enumerate(self.all_page_data):
                        if pd is None:
                            smart_idx = i
                            break
                        en = pd.get('english', '')
                        if "[Translation Pending]" in en or "[Translation Error]" in en:
                            smart_idx = i
                            break
                    
                    self.current_page_idx = smart_idx
                    self.schedule_ui_update(lambda: self.render_page(self.current_page_idx))
                    self.schedule_ui_update(lambda: self.page_label.configure(text=f"Page {self.current_page_idx + 1} of {self.total_pages}"))
                    
            except Exception as e:
                self.schedule_ui_update(lambda msg=str(e): messagebox.showerror("Resume Error", msg))
            finally:
                self.schedule_ui_update(lambda: self.status_label.configure(text="Ready"))
                
        threading.Thread(target=load_cache, daemon=True).start()

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.load_new_pdf(file_path)

    def load_new_pdf(self, file_path):
        """Standardizes the process of loading a fresh PDF from a path (Manual or Drag-and-Drop)."""
        if not file_path: return
        try:
            self.close_book() # Cleanup old session properly
        except Exception as e:
            print(f"Error closing previous book: {e}")
            
        self.schedule_ui_update(lambda: self.status_label.configure(text="Processing..."))
        
        # Clear previous entries visually from the recycled ui
        self.schedule_ui_update(lambda: self.page_number_label.configure(text="Processing new document..."))
        
        if getattr(self, "supabase", None):
            self.publish_btn.configure(state="normal")
            self.sync_btn.configure(state="normal")
            
        self.reset_ui(hard_reset=False)
        self.visual_img_label.configure(image=None, text="[Loading...]")
        self.stop_requested = True
        if getattr(self, "active_executor", None):
            try: 
                self.active_executor.shutdown(wait=False, cancel_futures=True)
            except: 
                pass
            self.active_executor = None
            
        self.current_pdf_path = file_path
        self.cache_dir = get_app_path(".translator_cache")
        # Start processing logic...
        if self.ui_pdf_doc:
            try: self.ui_pdf_doc.close()
            except: pass
            self.ui_pdf_doc = None
        
        self.current_session += 1
        self.current_pdf_path = file_path
        self.stop_requested = False
        
        # Reset state for new book
        self.all_page_data = []
        self.image_cache = {}
        self.current_page_idx = 0
        self.total_pages = 0
        # Start everything in background to keep UI fluid
        threading.Thread(target=self.initial_upload_kickoff, args=(file_path,), daemon=True).start()

    def start_fresh_processing(self, file_path):
        """Clears local cache and starts OCR pipeline."""
        book_name = os.path.basename(file_path).replace(".pdf", "")
        self.cache_dir = get_app_path(os.path.join(".translator_cache", book_name))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.save_settings() # SPRINT 56: Save сразу после старта новой книги для корректного Resume
        
        # 2. Cache Cleanup (Disk I/O)
        import glob
        for f in glob.glob(os.path.join(self.cache_dir, "*")):
            try: 
                if os.path.isfile(f): os.remove(f)
            except: pass
            
        # 3. Start OCR
        self.process_pdf(file_path)

    def change_page(self, direction):
        if not self.all_page_data:
            return
            
        # FORCE IMMEDIATE SAVE OF CURRENT PAGE BEFORE NAVIGATING AWAY
        self.save_edits_to_cache()
        
        new_idx = self.current_page_idx
        if direction == "first":
            new_idx = 0
        elif direction == "last":
            new_idx = len(self.all_page_data) - 1
        elif direction == "next":
            if self.current_page_idx < len(self.all_page_data) - 1:
                new_idx += 1
        elif direction == "prev":
            if self.current_page_idx > 0:
                new_idx -= 1
        elif direction == "jump":
            try:
                page_num = int(self.jump_entry.get())
                if 1 <= page_num <= len(self.all_page_data):
                    new_idx = page_num - 1
            except:
                pass
                
        if new_idx != self.current_page_idx or direction == "jump":
            print(f"[Nav] Moving to Index {new_idx} (Page {new_idx+1})")
            self.current_page_idx = int(new_idx)
            self.render_page(self.current_page_idx)

    def save_page_to_cache(self, page_data):
        # Save images separately if they exist to keep RAM lean
        page_num = page_data['page']
        cache_path = os.path.join(self.cache_dir, f"page_{page_num}.json")
        
        # Strip PIL images from JSON, save them as PNGs
        save_data = page_data.copy()
        if save_data.get('cover_image'):
            img_path = os.path.join(self.cache_dir, f"img_{page_num}.jpg")
            save_data['cover_image'].save(img_path)
            save_data['cover_image_path'] = img_path
            # We explicitly delete it from the original page_data dictionary in memory!
            # The render_page function will lazy load it back if needed
            del save_data['cover_image']
            page_data['cover_image_path'] = img_path
            
        # Add the original PDF path to the cache
        if self.current_pdf_path:
            save_data['original_pdf_path'] = self.current_pdf_path
            page_data['original_pdf_path'] = self.current_pdf_path
            
        import json
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            
        # Add to Sync Queue (sequential to save RAM)
        if self.supabase:
            with self.sync_lock:
                self.sync_queue.append(save_data)
                if not self.sync_worker_active:
                    self.sync_worker_active = True
                    threading.Thread(target=self.process_sync_queue, daemon=True).start()

    def show_reader_context_menu(self, event):
        """Shows a custom right-click context menu with 'Magic' reader actions."""
        widget = event.widget
        menu = tk.Menu(self, tearoff=0)
        
        has_selection = False
        selected_text = ""
        try:
            selected_text = widget.get("sel.first", "sel.last").strip()
            if selected_text:
                has_selection = True
        except tk.TclError:
            pass # No selection
        
        # --- Standard Actions ---
        menu.add_command(label="✂ Cut", accelerator="Ctrl+X",
                         command=lambda: widget.event_generate("<<Cut>>"),
                         state="normal" if has_selection else "disabled")
        menu.add_command(label="📋 Copy", accelerator="Ctrl+C",
                         command=lambda: widget.event_generate("<<Copy>>"),
                         state="normal" if has_selection else "disabled")
        menu.add_command(label="📥 Paste", accelerator="Ctrl+V",
                         command=lambda: widget.event_generate("<<Paste>>"))
        
        menu.add_separator()
        
        # --- Magic Actions (require selection) ---
        menu.add_command(label="🔍 Look Up Selection",
                         command=lambda: self._reader_lookup(selected_text, event),
                         state="normal" if has_selection else "disabled")
        menu.add_command(label="🤖 Re-Translate Snippet",
                         command=lambda: self._reader_retranslate(selected_text, widget, event),
                         state="normal" if has_selection else "disabled")
        menu.add_command(label="📝 Add to Glossary",
                         command=lambda: self._reader_add_glossary(selected_text),
                         state="normal" if has_selection else "disabled")
        
        menu.add_separator()
        menu.add_command(label="📤 Send to Quick Translate",
                         command=lambda: self._reader_send_to_qt(selected_text),
                         state="normal" if has_selection else "disabled")
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def show_image_context_menu(self, event, mode="translation"):
        """Shows a custom right-click context menu for images."""
        menu = tk.Menu(self, tearoff=0)
        
        if mode == "translation":
            menu.add_command(label="🔍 Re-Scan This Page", command=self.rescan_current_page)
            menu.add_command(label="📤 Send to Quick Translate", command=self._image_send_to_qt)
            menu.add_separator()
            menu.add_command(label="📥 Save Image As...", command=self._image_save_as)
        else: # quick translate mode
            menu.add_command(label="🔄 Re-Run Image Translation", command=self.run_image_translate)
            menu.add_command(label="🗑️ Clear Image", command=self._image_clear_qt)
            
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def reset_ui(self, hard_reset=False):
        """SPRINT 79: Standardized UI clearing to avoid stale data between sessions."""
        def clear_box(box):
            if box:
                try:
                    box.configure(state="normal")
                    box.delete("1.0", "end")
                    box.configure(state="disabled")
                except: pass
        
        clear_box(self.orig_text)
        clear_box(self.lit_text)
        clear_box(self.trans_text)
        
        if hasattr(self, "visual_img_label"):
             self.visual_img_label.configure(image=None, text="[No Document Loaded]")
             self.visual_img_label.pack_forget() # Hide visual feed by default
             
        if hard_reset:
            self.schedule_ui_update(lambda: self.page_number_label.configure(text="Page 0"))
            self.schedule_ui_update(lambda: self.status_label.configure(text="Ready."))
        
    def save_edits_to_cache(self):
        """Pulls the current text from all 3 boxes and rewrites the cache JSON."""
        if not self.all_page_data or self.current_page_idx >= len(self.all_page_data):
            return
            
        # SPRINT 33 Fix: If the page is currently processing, don't try to save edits to it
        page_data = self.all_page_data[self.current_page_idx]
        if page_data is None or 'is_image' not in page_data:
            return
            
        # Get live text (ignoring the trailing newline Tkinter always adds)
        new_orig = self.orig_text.get("1.0", "end-1c")
        new_lit = self.lit_text.get("1.0", "end-1c")
        new_trans = self.trans_text.get("1.0", "end-1c")
        
        # Update memory struct
        self.all_page_data[self.current_page_idx]['original'] = new_orig
        self.all_page_data[self.current_page_idx]['literal'] = new_lit
        self.all_page_data[self.current_page_idx]['english'] = new_trans
        
        # Save straight to disk
        self.save_page_to_cache(self.all_page_data[self.current_page_idx])

    def render_page(self, page_idx):
        try:
            # SPRINT 82: Harmonize global index with actual rendered page
            # This fixes the "always sends page 1 to QT" bug during background processing
            self.current_page_idx = int(page_idx)
            
            if not self.all_page_data or page_idx >= len(self.all_page_data): return
            data = self.all_page_data[page_idx]
            
            # SPRINT 11.1: Update footer label IMMEDIATELY before potential early returns
            total = len(self.all_page_data)
            self.page_label.configure(text=f"Page {page_idx + 1} of {total}")
            self.page_number_label.configure(text=f"Page {page_idx + 1}")
            
            # Handle Placeholder (Not processed yet) or Poisoned Cache
            if data is None or 'original' not in data or data.get('original') == "[Page Still Processing...]":
                # SPRINT 45: Instant Content Load (Show image before OCR)
                img_ref = None
                if self.current_pdf_path:
                    try:
                        if not getattr(self, "ui_pdf_doc", None):
                            self.ui_pdf_doc = fitz.open(self.current_pdf_path)
                        page = self.ui_pdf_doc.load_page(page_idx)
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                        img_ref = Image.open(io.BytesIO(pix.tobytes("png")))
                    except:
                        self.ui_pdf_doc = None

                self.populate_ui_skeleton(
                    page_num=page_idx, 
                    original="[Page Still Processing...]", 
                    translated="Please wait for OCR to complete.", 
                    literal="[Processing...]", 
                    is_image=False, 
                    is_rtl_override=False, 
                    is_centered=True, 
                    is_cover=False, 
                    cover_image=img_ref
                )
                return

            page_id = data['page']
            
            # SPRINT 11.1: Lazy Image Fetching for Cloud Books
            img_ref = self.image_cache.get(page_id)
            if not img_ref and data.get('cover_image_path') and os.path.exists(data['cover_image_path']):
                try: img_ref = Image.open(data['cover_image_path'])
                except: pass

            if not img_ref and data.get('cloud_id') and not os.path.exists(data.get('cover_image_path', '')):
                def _fetch_page_img():
                    try:
                        c_id = data['cloud_id']
                        p_idx = data['page']
                        img_path = data['cover_image_path']
                        # Immediate feedback
                        self.schedule_ui_update(lambda: self.status_label.configure(text=f"Fetching page {p_idx+1} image..."))
                        
                        # Storage Download
                        res_img = self.supabase.storage.from_("book-images").download(f"{c_id}/img_{p_idx}.jpg")
                        with open(img_path, 'wb') as f:
                            f.write(res_img)
                        
                        # Refresh UI if we are still on this page
                        if self.current_page_idx == p_idx:
                            self.schedule_ui_update(lambda: self.render_page(p_idx))
                    except Exception as e:
                        print(f"Lazy fetch failed for page {p_idx}: {e}")
                
                threading.Thread(target=_fetch_page_img, daemon=True).start()
                # Show loading skeleton while waiting
                self.populate_ui_skeleton(
                    page_num=page_idx, 
                    original=data.get('original', ''), 
                    translated=data.get('english', ''), 
                    literal=data.get('literal', ''), 
                    is_image=data.get('is_image', False), 
                    is_rtl_override=data.get('is_rtl_page'), 
                    is_centered=data.get('is_centered', False), 
                    is_cover=data.get('is_cover', False),
                    cover_image=None # Triggers "[Loading Image...]" visual
                )
                return


            if img_ref:
                self.image_cache[page_id] = img_ref
                if len(self.image_cache) > 5:
                    oldest_key = next(iter(self.image_cache))
                    self.image_cache.pop(oldest_key, None)

            self.populate_ui_skeleton(
                page_num=data['page'], 
                original=data['original'], 
                translated=data['english'], 
                literal=data['literal'], 
                is_image=data.get('is_image', False), 
                is_rtl_override=data.get('is_rtl_page', False), 
                is_centered=data.get('is_centered', False), 
                is_cover=data.get('is_cover', False), 
                cover_image=img_ref
            )
            
            self.page_label.configure(text=f"Page {page_idx + 1} of {len(self.all_page_data)}")
        except Exception as e:
            print(f"[Critical Render Crash] Page {page_idx}: {e}")
        self.update_bookmark_ui()

    def toggle_pause(self):
        if not self.is_paused:
            # We are pausing
            self.is_paused = True
            self.stop_requested = True
            self.status_label.configure(text="Pausing...")
            self.pause_btn.configure(text="Resume", fg_color="#2E7D32", hover_color="#1B5E20")
        else:
            # We are resuming
            self.is_paused = False
            self.stop_requested = False
            self.pause_btn.configure(text="Pause", fg_color="#C62828", hover_color="#B71C1C")
            self.status_label.configure(text="Resuming translation...")
            self.menu_bar.entryconfigure("Export", state="disabled") 

            
            # Re-bind the click so it can be paused again
            self.pause_btn.configure(command=self.toggle_pause)
            
            # Find the first page that still needs processing
            start_page = 0
            if self.all_page_data:
                for i, data in enumerate(self.all_page_data):
                    if data is None:
                        start_page = i
                        break
                    # Also catch pages restored from cache that still need translation
                    en = data.get('english', '') if isinstance(data, dict) else ''
                    if not en or "[Translation Pending]" in en or "[Translation Error]" in en or "[Translation Failed]" in en:
                        start_page = i
                        break
                else:
                    start_page = len(self.all_page_data)  # Truly all done
            
            if self.current_pdf_path:
                threading.Thread(target=self.process_pdf, args=(self.current_pdf_path, start_page), daemon=True).start()

    def get_valid_pdf_path(self):
        if self.current_pdf_path and os.path.exists(self.current_pdf_path):
            return True
        # If missing, ask the user
        file_path = filedialog.askopenfilename(title="Select Original PDF to Re-Scan", filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.current_pdf_path = file_path
            # optionally save to cache
            if self.all_page_data:
                self.all_page_data[0]['original_pdf_path'] = file_path
                self.save_page_to_cache(self.all_page_data[0])
            return True
        return False

    def rescan_current_page(self):
        if not self.all_page_data: return
        if not self.get_valid_pdf_path(): return
        
        self.rescan_btn.configure(state="disabled")
        if hasattr(self, 'rescan_err_btn'): self.rescan_err_btn.configure(state="disabled")
        threading.Thread(target=self.process_pdf, args=(self.current_pdf_path, self.current_page_idx, True), daemon=True).start()

    def rescan_all_errors(self):
        if not self.all_page_data: return
        if not self.get_valid_pdf_path(): return
        
        pages_to_rescan = []
        for i, pd in enumerate(self.all_page_data):
            en = pd.get('english', '')
            lit = pd.get('literal', '')
            if "[Translation Error]" in en or "[OCR Trans Error:" in en or "[Layout Error:" in en or "[Sync Error]" in lit:
                pages_to_rescan.append(i)
                
        if not pages_to_rescan:
            messagebox.showinfo("Re-Scan", "No translation errors found!")
            return
            
        self.rescan_btn.configure(state="disabled")
        self.rescan_err_btn.configure(state="disabled")
        threading.Thread(target=self.process_pdf, args=(self.current_pdf_path, 0, False, pages_to_rescan), daemon=True).start()

    def process_pdf(self, file_path, start_page=0, single_page_mode=False, pages_to_process=None):
        try:
            # CRITICAL: Align self.cache_dir with the book-specific subfolder
            # This MUST match where ocr_worker saves images (line ~184)
            book_name = os.path.basename(file_path).replace(".pdf", "")
            self.cache_dir = get_app_path(os.path.join(".translator_cache", book_name))
            os.makedirs(self.cache_dir, exist_ok=True)
            
            session_at_start = self.current_session
            self.stop_requested = False
            self.is_paused = False
            if not single_page_mode and not pages_to_process:
                self.schedule_ui_update(lambda: self.pause_btn.configure(state="normal", text="Pause", fg_color="#C62828", hover_color="#B71C1C", command=self.toggle_pause))
                self.schedule_ui_update(lambda: self.menu_bar.entryconfigure("Export", state="disabled"))
            else:
                self.schedule_ui_update(lambda: self.status_label.configure(text=f"Rescanning..."))
                
            self.schedule_ui_update(lambda: self.status_label.configure(text="Opening PDF..."))
            doc = fitz.open(file_path)
            num_pages_to_process = len(doc)
            self.schedule_ui_update(lambda: self.status_label.configure(text=f"PDF Opened ({num_pages_to_process} pages)"))
            
            # --- PRE-ALLOCATE Order-Aware Slots ---
            if not pages_to_process:
                # SPRINT 33 FIX: Do not wipe existing cache if we are resuming
                if not self.all_page_data:
                    self.all_page_data = [{"page": i} for i in range(num_pages_to_process)]
                elif len(self.all_page_data) < num_pages_to_process:
                    self.all_page_data.extend([{"page": i} for i in range(len(self.all_page_data), num_pages_to_process)])
                elif len(self.all_page_data) > num_pages_to_process:
                    self.all_page_data = self.all_page_data[:num_pages_to_process]
                
                self.total_pages = num_pages_to_process
                self.schedule_ui_update(lambda: self.page_label.configure(text=f"Page {self.current_page_idx + 1} of {num_pages_to_process}"))
                self.schedule_ui_update(lambda: self.render_page(self.current_page_idx)) # Show current slot immediately

            selected_lang = self.lang_menu.get()
            is_auto = (selected_lang == "Auto-Detect")
            
            page_rect = doc[0].rect
            page_width = page_rect.width

            if self.active_executor:
                try: self.active_executor.shutdown(wait=False, cancel_futures=True)
                except: pass
                self.active_executor = None
            
            # Ensure translation executor exists
            if not self.translation_executor:
                self.translation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
            
            ocr_lang_code = self.languages[selected_lang]["ocr"]
            src_lang_code = self.languages[selected_lang]["trans"]
            is_rtl = False
            
            translator_src = 'auto' if is_auto else src_lang_code
            translator = GoogleTranslator(source=translator_src, target='en')

            loop_iterable = pages_to_process if pages_to_process else range(start_page, num_pages_to_process if not single_page_mode else min(start_page + 1, num_pages_to_process))
            num_total = len(loop_iterable)

            if self.speed_var.get() and len(loop_iterable) > 1 and not single_page_mode:
                # MULTI-PROCESS PIPELINE
                self.schedule_ui_update(lambda: self.status_label.configure(text=f"Pipelined OCR (Parallel)..."))
                
                # Worker count: Balanced (50% cores) OR Eco (2 cores)
                cpu_count = os.cpu_count() or 4
                max_workers = int(2 if self.eco_var.get() else max(1, cpu_count // 2))
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    self.active_executor = cast(Any, executor)
                    
                    # SPRINT 73: Streaming Pipeline bounds memory on huge PDFs
                    page_iter = iter(loop_iterable)
                    future_to_page = {}
                    
                    # Prime the pump with initial batch
                    for _ in range(max_workers * 2):
                        try:
                            p_num = next(page_iter)
                            if self.stop_requested or self.current_session != session_at_start: break
                            f = executor.submit(ocr_worker, p_num, self.current_pdf_path, ocr_lang_code, is_auto, self.languages, translator_src, page_width, is_rtl, self.tesseract_path)
                            future_to_page[f] = p_num
                        except StopIteration:
                            break
                            
                    while future_to_page:
                        if self.stop_requested or self.current_session != session_at_start: break
                        
                        done, _ = concurrent.futures.wait(future_to_page.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                        
                        for f in done:
                            p_num_expected = future_to_page.pop(f)
                            
                            self.schedule_ui_update(lambda i=p_num_expected+1: self.status_label.configure(text=f"Analyzing Page {i}..."))
                            
                            try:
                                res = f.result(timeout=120)
                                p_num_returned = int(cast(Any, res).get('page', -1)) if res and isinstance(res, dict) else -1
                                print(f"[MainThread] Received result for Page {p_num_expected} (Worker reported: {p_num_returned})")
                                
                                if res and isinstance(res, dict) and "error" not in res:
                                    p_num = int(cast(Any, res).get('page', 0))
                                    if p_num != p_num_expected:
                                        res['page'] = p_num_expected
                                        p_num = p_num_expected
                                    if p_num < len(self.all_page_data):
                                        if self.all_page_data[p_num] is None:
                                            self.all_page_data[p_num] = res
                                        else:
                                            self.all_page_data[p_num].update(res)
                                    self.save_page_to_cache(res)
                                    
                                    if p_num == self.current_page_idx and self.current_session == session_at_start:
                                        self.schedule_ui_update(lambda idx=p_num: self.render_page(int(idx)))
                                    
                                    source_text = str(res.get('original', ''))
                                    src_lang = str(res.get('translator_src', 'auto'))
                                    
                                    if len(source_text) > 5 and "[No Text" not in source_text:
                                        context_data = None
                                        if self.selected_engine == "GPT-4o":
                                            context_data = {
                                                "prev_page": self.all_page_data[p_num-1].get('original', '') if p_num > 0 else "",
                                                "next_page": self.all_page_data[p_num+1].get('original', '') if p_num < len(self.all_page_data)-1 else ""
                                            }
                                        f_trans = self.translation_executor.submit(
                                            translate_worker, 
                                            text=source_text, 
                                            translator_src=src_lang, 
                                            engine=self.selected_engine, 
                                            deepl_key=self.deepl_key, 
                                            openai_key=self.openai_key, 
                                            ollama_model=getattr(self, 'ollama_model', 'llama3'),
                                            glossary=self.glossary, 
                                            context=context_data
                                        )
                                        
                                        def on_trans_done(fut, page_idx=p_num, sess=session_at_start):
                                            try:
                                                t_res = fut.result()
                                                if "error" not in t_res and self.current_session == sess:
                                                    self.all_page_data[page_idx]['english'] = t_res['translated']
                                                    self.all_page_data[page_idx]['literal'] = t_res['literal']
                                                    self.save_page_to_cache(self.all_page_data[page_idx])
                                                    if page_idx == self.current_page_idx:
                                                        self.schedule_ui_update(lambda i=page_idx: self.render_page(int(i)))
                                                elif "error" in t_res:
                                                    self.all_page_data[page_idx]['english'] = f"[Translation Failed: {str(t_res.get('error', 'Unknown'))[:100]}]"
                                                    self.save_page_to_cache(self.all_page_data[page_idx])
                                                    if page_idx == self.current_page_idx:
                                                        self.schedule_ui_update(lambda i=page_idx: self.render_page(int(i)))
                                            except Exception as te:
                                                if page_idx < len(self.all_page_data):
                                                    self.all_page_data[page_idx]['english'] = f"[Error: {str(te)[:100]}]"
                                                    self.save_page_to_cache(self.all_page_data[page_idx])
                                                    if page_idx == self.current_page_idx:
                                                        self.schedule_ui_update(lambda i=page_idx: self.render_page(int(i)))
                                            finally:
                                                if self.current_session == sess:
                                                    self.schedule_ui_update(lambda i=page_idx+1: self.status_label.configure(text=f"Finished Page {i}"))
                                            
                                        f_trans.add_done_callback(on_trans_done)
                                    else:
                                        self.all_page_data[p_num]['english'] = ""
                                        self.schedule_ui_update(lambda i=p_num+1: self.status_label.configure(text=f"Finished Page {i} (Image Only)"))
                                    
                                    self.schedule_ui_update(self.update_idletasks)
                                else:
                                    if res and "error" in res: raise Exception(res["error"])
                                    else: raise Exception("Worker returned empty result")
                            except Exception as e:
                                print(f"[Parallel Task Failed] Page {p_num_expected}: {e}")
                                err_data = {
                                    "page": p_num_expected,
                                    "original": f"[Error on Page {p_num_expected + 1}]",
                                    "english": f"Processing failed: {str(e)[:200]}",
                                    "literal": "",
                                    "is_image": False,
                                    "is_rtl_page": False,
                                    "is_centered": False,
                                    "is_cover": False
                                }
                                if p_num_expected < len(self.all_page_data):
                                    self.all_page_data[p_num_expected] = err_data
                                    if p_num_expected == self.current_page_idx:
                                        self.schedule_ui_update(lambda v=p_num_expected: self.render_page(int(v)))
                                        
                            # Queue the next page so we never exceed our sliding window
                            try:
                                next_p = next(page_iter)
                                if not self.stop_requested and self.current_session == session_at_start:
                                    new_f = executor.submit(ocr_worker, next_p, self.current_pdf_path, ocr_lang_code, is_auto, self.languages, translator_src, page_width, is_rtl, self.tesseract_path)
                                    future_to_page[new_f] = next_p
                            except StopIteration:
                                pass
                        
            else:
                # SEQUENTIAL LOOP
                for page_num in loop_iterable:
                    if self.stop_requested or self.current_session != session_at_start: break
                    if self.is_paused:
                        while self.is_paused and not self.stop_requested: time.sleep(0.5)
                    self.process_single_page(doc, page_num, translator, ocr_lang_code, is_auto, selected_lang, is_rtl, single_page_mode, pages_to_process, num_pages_to_process)
                    # SPRINT 27: Stealth heartbeat (3.0s delay)
                    time.sleep(3.0 + random.uniform(0.1, 0.5))

            if not self.stop_requested:
                if pages_to_process:
                    self.schedule_ui_update(lambda: self.status_label.configure(text="Batch Re-scan Complete!"))
                elif single_page_mode:
                    self.schedule_ui_update(lambda: self.status_label.configure(text="Page Re-scan Complete!"))
                else:
                    self.schedule_ui_update(lambda: self.status_label.configure(text="Complete!"))
                    self.schedule_ui_update(lambda: self.pause_btn.configure(state="disabled"))
                    
                self.schedule_ui_update(lambda: self.menu_bar.entryconfigure("Export", state="normal"))
                self.schedule_ui_update(lambda: self.publish_btn.configure(state="normal"))
            
            if 'doc' in locals() and doc:
                doc.close() # Ensure closure
            
            self.active_executor = None
            import gc
            gc.collect() # Force cleanup
            
        except Exception as e:
            msg = humanize_error(e)
            self.schedule_ui_update(lambda e_msg=msg: messagebox.showerror("Error", e_msg))
        finally:
             self.schedule_ui_update(lambda: self.status_label.configure(text="Ready"))
             if single_page_mode or pages_to_process:
                 self.schedule_ui_update(lambda: self.rescan_btn.configure(state="normal"))
                 if hasattr(self, 'rescan_err_btn'): self.schedule_ui_update(lambda: self.rescan_err_btn.configure(state="normal"))

    def clean_ocr_text(self, text):
        if not text: return ""
        
        # SPRINT 33: "Clean" Toggle Logic
        # If the user turns ON the Clean switch, we aggressively hunt down page-border noise
        if hasattr(self, 'deep_cleanup_var') and self.deep_cleanup_var.get():
            # 1. Strip common isolated noise symbols and border fragments that OCR hallucinates
            # TARGET NOISE: ~ ^ | _ \ / ` 
            # We use word boundaries / spacing to ensure we don't delete math equations or inline brackets
            
            # Remove isolated symbols (surrounded by spaces)
            text = re.sub(r'(?<= )[\~\|\_\^\/\\](?= )', '', text)
            
            # 2. Remove repetitive bridge characters (borders)
            # e.g., ".......", "-------", "_____"
            text = re.sub(r'[\.\-\_\=]{4,}', ' ', text)
            
            # 3. Heuristic Filter: Remove "words" that are mostly symbols or numbers
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                words = line.split()
                valid_words = []
                for word in words:
                    # If word has at least 1 letter or is a common number/punc, keep it
                    letter_count = len(re.findall(r'[a-zA-Z\u00C0-\u017F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]', word))
                    if letter_count > 0 or word.isnumeric() or len(word) < 4:
                        valid_words.append(word)
                
                cleaned_line = " ".join(valid_words).strip()
                # Filter lines that are just single random characters (often border noise)
                if len(cleaned_line) > 1:
                    cleaned_lines.append(cleaned_line)
                    
            text = "\n".join(cleaned_lines)
            
        return text


    def rebalance_scripts(self, text):
        """Fixes common French/Greek script confusion and bracket mangling."""
        if not text: return ""
        
        # 1. Fix French 'le' becoming Greek 'ἰο' or 'ἰο'
        # Tesseract often confuses 'le' with 'ἰo' or 'ἰο' (Greek iota-omicron)
        # We look for Greek snippets that are likely French words
        text = re.sub(r'\bἰ[οo]\b', 'le', text)
        text = re.sub(r'\bἰ[δθ]\b', 'le', text) # Alternative mangling
        
        # 2. Fix Greek 'αο' becoming French 'que' or 'ce'
        # Scholarship text: 'αο' (Greek alpha-omicron) is often 'que' or 'ce' or 'ou'
        # If surrounded by French latin text, these are almost certainly errors
        # (This is a subtle fix, we only apply to very common small errors)
        text = re.sub(r' ([αα]ο|οα) ', ' le ', text)
        
        # 3. Bracket mangling cleanup
        # Fix (( into (
        text = re.sub(r'\({2,}', '(', text)
        text = re.sub(r'\){2,}', ')', text)
        # Fix ((. into (f. (Common for folio markers in intro)
        text = re.sub(r'\(\(\. ', '(f. ', text)
        # Fix 6"-7 to 6-7
        text = re.sub(r'6\"-7', '6-7', text)
        
        # 4. Fix Greek iota mangling in French words
        text = re.sub(r'\bἰο\b', 'le', text)
        text = re.sub(r'ἰο tome', 'le tome', text)
        
        return text

    def tokenize_text(self, text: str) -> list:
        # Unified regex for word/number detection
        return re.findall(r'\b\w+\b', str(text))

    def populate_ui_skeleton(self, page_num, original, translated, literal, is_image, is_rtl_override=None, is_centered=False, is_cover=False, cover_image=None):
        """Updates the permanent UI skeleton with the current page data instantly."""
        # Pic 2/3 Clean Look: Hide "No Text" labels
        display_orig = "" if "[No Text Detected]" in str(original) else original
        display_trans = "" if "[No Text Detected]" in str(translated) or "[Translation Pending]" in str(translated) else translated
        if "[Translation Pending]" in str(translated): display_trans = "[Translation Pending]"

        for box in [self.orig_text, self.trans_text, self.lit_text]:
            box.configure(state="normal")
            box.delete("1.0", "end")
            if getattr(self, "read_only_mode", False):
                box.configure(state="disabled")
        self.visual_img_label.configure(image="")

        is_rtl = is_rtl_override if is_rtl_override is not None else (self.lang_menu.get() in ["Arabic", "Hebrew", "Yiddish"])
        
        # UI is 1-based, internal is 0-based
        self.page_number_label.configure(text=f"Page {page_num + 1}")

        # Pic 1: Uniformity. All pages now show the 3-column layout.
        show_text_panels = True 
        
        if show_text_panels:
            self.orig_container.grid(row=1, column=1, sticky="nsew")
            self.trans_container.grid(row=1, column=3, sticky="nsew")
            self.mid_container.grid(row=1, column=2, columnspan=1, sticky="nsew")
            
            # Transcription Column
            if is_rtl:
                self.orig_text.tag_config("rtl", justify="right")
                self.insert_and_tag_words(self.orig_text, display_orig, is_rtl=True)
            else:
                self.insert_and_tag_words(self.orig_text, display_orig)
                
            # Translation Column
            self.insert_and_tag_words(self.trans_text, display_trans, is_centered=is_centered)
        else:
            # ONLY Page 0 (Cover) shows as Image Only
            self.orig_container.grid_remove()
            self.trans_container.grid_remove()
            self.mid_container.grid(row=1, column=0, columnspan=4, sticky="nsew")

        # Middle Column (1-1 Match OR Image feed)
        if self.one_one_var.get() and show_text_panels:
            self.visual_img_label.pack_forget()
            self.lit_text.pack(padx=15, pady=15, fill="both", expand=True)
            self.insert_and_tag_words(self.lit_text, literal)
        else:
            self.lit_text.pack_forget()
            self.visual_img_label.pack(padx=5, pady=25, fill="both", expand=True, anchor="n")
            
            img_to_show = cover_image
            if not img_to_show and isinstance(original, Image.Image):
                img_to_show = original
            
            if img_to_show:
                display_img = img_to_show.copy()
                max_w, max_h = (500, 800)
                display_img.thumbnail((max_w, max_h))
                tk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=(display_img.width, display_img.height))
                self.visual_img_label.configure(image=tk_img, text="")
                self.visual_img_label._image = tk_img # Keep reference
            else:
                self.visual_img_label.configure(image=None, text="[No Image Capture]")

    def insert_and_tag_words(self, textbox, text, is_rtl=False, is_centered=False):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        
        # Simple insertion without word-level highlighting for clarity
        clean_text = str(text)
        textbox.insert("1.0", clean_text)
        
        if is_rtl:
            textbox.tag_add("content", "1.0", "end")
            textbox.tag_config("content", justify="right")
        if is_centered:
            textbox.tag_add("content", "1.0", "end")
            textbox.tag_config("content", justify="center")
            
        # SPRINT 11.1: Standardize lock enforcement after every insertion
        if getattr(self, "read_only_mode", False):
            textbox.configure(state="disabled")

    def link_highlighting(self, textboxes):
        pass # Highlighting disabled as requested

