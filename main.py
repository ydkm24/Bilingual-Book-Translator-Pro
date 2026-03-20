# === THE GREAT REFACTOR ===
# main.py — Application entry point and class orchestrator.
# Utilities, workers, and mixins are imported from their respective modules.

import os
import sys
import threading
import time
import random
import multiprocessing
import concurrent.futures
from functools import partial
from typing import Any, Dict, List, Optional, cast
import requests
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageTk
import io
import re
import cv2
import json
import numpy as np
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import LETTER
from supabase import create_client, Client

# Import extracted modules
from utils import (
    APP_VERSION, humanize_error, resource_path, get_app_path,
    PAPER_BG, TEXT_COLOR, TEXT_DIM, HEADER_BG, ACCENT_COLOR, ACCENT_HOVER,
    PAPER_SHEET_BG, BORDER_COLOR, CARD_BG, FRAME_BG,
    BTN_PRIMARY, BTN_PRIMARY_HOVER, BTN_SECONDARY, BTN_SECONDARY_HOVER,
    INPUT_BG, BTN_DANGER, BTN_DANGER_HOVER, BTN_SUCCESS, BTN_SUCCESS_HOVER,
    BTN_WARNING, BTN_WARNING_HOVER,
)

# Set appearance mode and color theme
if __name__ == "__main__":
    ctk.set_appearance_mode("Dark") 
    ctk.set_default_color_theme("blue")

from workers import ocr_worker, translate_worker
from cloud import CloudMixin
from export import ExportMixin
from processing import ProcessingMixin
from ui_components import UIComponentsMixin


class PDFTranslatorApp(CloudMixin, ExportMixin, ProcessingMixin, UIComponentsMixin, ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bilingual Book Translator Pro")
        self.geometry("1450x900")
        self.configure(fg_color=PAPER_BG)

        # --- Base State & Configuration ---
        self.settings_path = get_app_path("settings.json")
        self.terminology_path = get_app_path("terminology.json")
        self.glossary = {} # Initialize before load_settings
        self.load_settings()
        
        # SPRINT 49: Cleanup zombie folders from v1.1.5/v1.1.6 bug
        for filename in ["session.json", "config.json"]:
            p = get_app_path(filename)
            if os.path.exists(p) and os.path.isdir(p):
                try: os.rmdir(p)
                except: pass
        
        # SPRINT 48: Pre-initialize Supabase attributes to satisfy early UI calls
        self.supabase = None
        self.current_user = None
        self.current_username = None
        self.current_book_id = ""
        
        # --- Language Mapping ---
        self.languages = {
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
            "Hebrew": {"ocr": "heb", "trans": "iw"}, # SPRINT 80 FIX: Google uses 'iw' legacy code
            "Greek": {"ocr": "ell", "trans": "el"},
            "Turkish": {"ocr": "tur", "trans": "tr"},
            "Dutch": {"ocr": "nld", "trans": "nl"},
            "Polish": {"ocr": "pol", "trans": "pl"},
            "Latin": {"ocr": "lat", "trans": "la"},
            "Yiddish": {"ocr": "yid", "trans": "yi"},
            "Auto-Detect": {"ocr": "eng", "trans": "auto"}
        }

        # Boolean Vars for Menu (Toggles)
        self.one_one_var = ctk.BooleanVar(value=False)
        self.deep_cleanup_var = ctk.BooleanVar(value=False)
        self.speed_var = ctk.BooleanVar(value=False)
        self.eco_var = ctk.BooleanVar(value=False)

        # Grid layout configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) 
        
        self.active_executor = None # type: Any
        self.translation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.ui_pdf_doc = None # Persistent handle for the UI thread
        self.current_session = 0 # Track session ID to isolate background tasks
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.grid_rowconfigure(0, weight=0) # Controls
        self.grid_rowconfigure(1, weight=0) # Headers
        self.grid_rowconfigure(2, weight=1) # Content
        
        # Top Frame for controls (Fixed height to prevent jitter from long titles)
        self.control_frame = ctk.CTkFrame(self, fg_color=HEADER_BG, corner_radius=0, height=45)
        self.control_frame.grid(row=0, column=0, sticky="ew")
        self.control_frame.pack_propagate(False)
        
        self.lang_label = ctk.CTkLabel(self.control_frame, text="Source Language:", text_color=TEXT_COLOR, font=("Arial", 12, "bold"))
        self.lang_label.pack(side="left", padx=(10, 5), pady=10)
        
        self.lang_menu = ctk.CTkOptionMenu(self.control_frame, values=list(self.languages.keys()), 
                                           fg_color=CARD_BG, button_color=CARD_BG, button_hover_color=BTN_SECONDARY_HOVER, width=120)
        self.lang_menu.pack(side="left", padx=5, pady=10)
        self.lang_menu.set("Auto-Detect")
        
        self.progress_bar = ctk.CTkProgressBar(self.control_frame, width=300, progress_color=ACCENT_COLOR)
        self.progress_bar.pack(side="left", padx=20, pady=10)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self.control_frame, text="Waiting for PDF...", text_color=TEXT_COLOR, 
                                        wraplength=250, anchor="w", justify="left")
        self.status_label.pack(side="left", padx=10, pady=5)

        # Avatar / Login Button (Far Right)
        self.avatar_btn = ctk.CTkButton(self.control_frame, text="👤 Login", width=80,
                                        fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR,
                                        command=self.handle_avatar_click)
        self.avatar_btn.pack(side="right", padx=10, pady=10)

        # Search bar (Right)
        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Search book...", 
                                         textvariable=self.search_var, width=150)
        self.search_entry.pack(side="right", padx=10, pady=10)
        self.search_entry.bind("<Return>", self.search_text)        

        # Action Buttons (Keep primary ones visible)
        self.sync_btn = ctk.CTkButton(self.control_frame, text="⇪ Sync", command=self.force_cloud_sync,
                                          width=60, fg_color=("#9B59B6", "#8E44AD"), hover_color=("#8E44AD", "#9B59B6"), state="disabled", text_color="#FFFFFF")
        # Note: sync_btn is NOT packed by default. It appears only for cloud books.

        self.publish_btn = ctk.CTkButton(self.control_frame, text="✨ Publish", command=self.publish_to_gallery,
                                          width=80, fg_color=("#FFB74D", "#FF9800"), hover_color=("#FFA726", "#F57C00"), state="disabled", text_color="#121212")
        self.publish_btn.pack(side="right", padx=5, pady=10)

        self.pause_btn = ctk.CTkButton(self.control_frame, text="Pause", command=self.toggle_pause,
                                      fg_color=("#E53935", "#C62828"), hover_color=("#D32F2F", "#B71C1C"), width=60, state="disabled", text_color="#FFFFFF")
        self.pause_btn.pack(side="right", padx=(5, 10), pady=10)
        
        # Settings Menu Dropdown Button
        self.settings_btn = ctk.CTkButton(self.control_frame, text="⚙", width=30, command=self.toggle_settings_popdown, fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR)
        self.settings_btn.pack(side="right", padx=5, pady=10)

        self.settings_popdown = None

        # --- UI: Top Menu Bar ---
        self.create_menu_bar()

        if self.supabase:
            self.publish_btn.configure(state="normal")
            self.sync_btn.pack_forget() # Hide Sync for new local upload
        else:
            self.publish_btn.configure(state="disabled")
            self.sync_btn.configure(state="disabled")
            
        # SPRINT 36.5: Sync the Top-Right UI with the pre-loaded authed session if one exists
        if hasattr(self, 'avatar_btn'):
            self.update_avatar_ui()
            
        # SPRINT 37: Check for updates on startup
        threading.Thread(target=self.check_for_updates, daemon=True).start()

        # SPRINT 47: Move Supabase/Auth to THE VERY END to avoid UI race conditions
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = None
        self.current_user = None # type: Optional[Dict[str, Any]]
        self.current_username = None # type: Optional[str]
        self.current_book_id = "" # Track cloud book ID
        
        if self.supabase_url and self.supabase_key and isinstance(self.supabase_url, str) and "YOUR_SUPABASE" not in self.supabase_url:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                
                # --- SPRINT 37: Session Auto-Login ---
                session_data = self.load_auth_session()
                if session_data and session_data.get('access_token'):
                    try:
                        res = self.supabase.auth.set_session(session_data['access_token'], session_data['refresh_token'])
                        if res and res.user and res.session:
                            self.current_user = {"id": res.user.id, "email": res.user.email}
                            self.save_auth_session(res.session) # SPRINT 49: Save refreshed tokens!
                            self.fetch_username()
                            self.update_avatar_ui()
                            self.refresh_library() 
                            print(f"Auto-login success: {res.user.email}")
                    except Exception as e:
                        # SPRINT 46: Be "Lazy" about clearing sessions.
                        err_str = str(e).lower()
                        is_token_error = any(x in err_str for x in ["invalid", "expired", "revoked", "rejected", "used", "consumed"])
                        if is_token_error:
                            print(f"Auto-login session invalid (clearing): {e}")
                            try: os.remove(get_app_path("session.json"))
                            except: pass
                        else:
                            print(f"Auto-login network error (keeping session): {e}")
            except Exception as e:
                print(f"Supabase Init Error: {e}")

        # Update controls based on Supabase presence
        if self.supabase:
            self.publish_btn.configure(state="normal")
            self.sync_btn.pack_forget() 
        else:
            self.publish_btn.configure(state="disabled")
            self.sync_btn.configure(state="disabled")

        self.update_avatar_ui()
        self.check_existing_session()
        
        # SPRINT 83: Quick Translate Stability
        self.qt_cancel_flag = False
        

    @property
    def cache_dir(self):
        return getattr(self, "_cache_dir", "")
        
    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value
        
    def apply_font_settings(self):
        font_tpl = (self.font_family, self.font_size)
        lit_font_tpl = (self.font_family, max(10, self.font_size - 1)) # Lit is slightly smaller
        
        self.orig_text.configure(font=font_tpl)
        self.trans_text.configure(font=font_tpl)
        self.lit_text.configure(font=lit_font_tpl)
        self.save_settings()

    def toggle_one_one(self):
        self.update_column_config()
        self.setup_headers()
        # Rerender current page to update layout (Visual Feed vs 1-1 Match)
        if self.all_page_data:
            self.render_page(self.current_page_idx)

    def update_column_config(self):
        # Sync weights between header and container
        # Always maintain 3 columns + index
        for area in [self.header_frame, self.main_container]:
            area.grid_columnconfigure(0, weight=1) # Page #
            area.grid_columnconfigure(1, weight=5) # Transcription
            area.grid_columnconfigure(2, weight=5) # 1-1 OR Visual Feed
            area.grid_columnconfigure(3, weight=5) # English
        
    def setup_headers(self):
        # Clear existing headers first to avoid stacking
        for widget in self.header_frame.winfo_children():
            widget.destroy()
            
        h_font = ("Inter", 14, "bold")
        ctk.CTkLabel(self.header_frame, text="#", font=h_font, text_color=ACCENT_COLOR).grid(row=0, column=0, pady=10, sticky="n")
        ctk.CTkLabel(self.header_frame, text="Transcription", font=h_font, text_color=TEXT_COLOR).grid(row=0, column=1, pady=10)
        
        # Dynamic Middle Header
        mid_text = "1-1 Match Pro" if self.one_one_var.get() else "Source Page"
        ctk.CTkLabel(self.header_frame, text=mid_text, font=h_font, text_color=ACCENT_COLOR).grid(row=0, column=2, pady=10)
            
        ctk.CTkLabel(self.header_frame, text="English Translation", font=h_font, text_color=TEXT_COLOR).grid(row=0, column=3, pady=10)
        
    def initial_upload_kickoff(self, file_path):
        """Heavy background lifting before starting full PDF processing."""
        self.schedule_ui_update(lambda: self.status_label.configure(text="Checking Cloud Match..."))
        
        # 1. Cloud Check (Network call - must be background)
        if self.supabase:
            book_title = os.path.basename(file_path)
            try:
                res = self.supabase.table("books").select("*").eq("title", book_title).execute()
                if res.data:
                    def ask_cloud():
                        use_cloud = messagebox.askyesno("Cloud Match Found", 
                                                        f"'{book_title}' has existing progress in the cloud. \n\nDo you want to pull cloud translations instead of starting fresh?")
                        if use_cloud:
                            self.load_book_from_cloud(res.data[0])
                        else:
                            # SPRINT 79: Explicitly background this so process_pdf doesn't lock UI
                            threading.Thread(target=self.start_fresh_processing, args=(file_path,), daemon=True).start()
                            
                    self.schedule_ui_update(ask_cloud)
                else:
                    # No match found, proceed to fresh processing (Already in BG thread)
                    self.start_fresh_processing(file_path)
            except Exception as e:
                print(f"Cloud check failed: {e}")
                self.start_fresh_processing(file_path)
        else:
            self.start_fresh_processing(file_path)

    def _reader_lookup(self, text, event):
        """Performs a quick dictionary-style lookup and shows a tooltip popup."""
        if not text:
            return
        
        def _job():
            try:
                translator = GoogleTranslator(source='auto', target='en')
                result = translator.translate(text)
                self.after(0, lambda: self._show_magic_tooltip(
                    f"🔍 \"{text}\"\n\n→ {result}", event))
            except Exception as e:
                self.after(0, lambda err=e: self._show_magic_tooltip(
                    f"❌ Lookup failed: {err}", event))
        
        threading.Thread(target=_job, daemon=True).start()

    def _reader_retranslate(self, text, widget, event):
        """Re-translates only the selected snippet using the current AI engine."""
        if not text:
            return
        
        self.status_label.configure(text="🤖 Re-translating snippet...")
        
        def _job():
            result = translate_worker(
                text=text, 
                translator_src='auto',
                target_lang=self.languages.get(self.lang_menu.get(), {"trans": "en"}).get("trans", "en"),
                engine=self.selected_engine,
                deepl_key=getattr(self, 'deepl_key', ''),
                openai_key=getattr(self, 'openai_key', ''),
                ollama_model=getattr(self, 'ollama_model', 'llama3'),
                glossary=self.glossary
            )
            
            translated = result.get('translated', result.get('error', 'Failed'))
            
            def _show():
                self._show_magic_tooltip(
                    f"🤖 Re-Translation ({self.selected_engine}):\n\n"
                    f"Original: \"{text[:80]}{'...' if len(text) > 80 else ''}\"\n\n"
                    f"→ {translated}", event)
                self.status_label.configure(text="Re-translation complete.")
            
            self.after(0, _show)
        
        threading.Thread(target=_job, daemon=True).start()

    def _reader_add_glossary(self, text):
        """Opens the Glossary Manager with the selected word pre-filled."""
        if not text:
            return
        
        gloss_win = ctk.CTkToplevel(self)
        gloss_win.title("Add to Glossary")
        gloss_win.geometry("420x280")
        gloss_win.attributes("-topmost", True)
        gloss_win.configure(fg_color=CARD_BG)
        
        ctk.CTkLabel(gloss_win, text="📝 Add Glossary Rule", 
                     font=("Inter", 18, "bold"), text_color=ACCENT_COLOR).pack(pady=(20, 10))
        ctk.CTkLabel(gloss_win, text="Ensure this word is always translated consistently.",
                     font=("Inter", 12), text_color=TEXT_DIM).pack(pady=(0, 20))
        
        form = ctk.CTkFrame(gloss_win, fg_color="transparent")
        form.pack(padx=30, fill="x")
        
        ctk.CTkLabel(form, text="Original Word:", font=("Inter", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        orig_entry = ctk.CTkEntry(form, width=250)
        orig_entry.grid(row=0, column=1, padx=10, pady=5)
        orig_entry.insert(0, text)
        
        ctk.CTkLabel(form, text="Always Translate As:", font=("Inter", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        target_entry = ctk.CTkEntry(form, width=250)
        target_entry.grid(row=1, column=1, padx=10, pady=5)
        target_entry.focus()
        
        def _save():
            orig = orig_entry.get().strip()
            target = target_entry.get().strip()
            if orig and target:
                self.glossary[orig] = target
                self.save_settings()
                self.status_label.configure(text=f"✅ Glossary rule added: {orig} → {target}")
                gloss_win.destroy()
        
        ctk.CTkButton(gloss_win, text="Save Rule", font=("Inter", 14, "bold"),
                      fg_color=BTN_SUCCESS, hover_color=BTN_SUCCESS_HOVER,
                      command=_save).pack(pady=20)

    def _reader_send_to_qt(self, text):
        """Sends the selected text to the Quick Translate input box."""
        if not text:
            return
        
        # Switch to Quick Translate tab
        self.tab_view.set("Quick Translate")
        self.qt_tabs.set("Text Mode")
        
        # Clear and insert
        self.qt_input.delete("1.0", "end")
        self.qt_input.insert("1.0", text)
        self.status_label.configure(text="📤 Text sent to Quick Translate.")

    def _show_magic_tooltip(self, text, event):
        """Shows a floating tooltip-style popup near the cursor with the result."""
        tip = ctk.CTkToplevel(self)
        tip.overrideredirect(True)
        tip.attributes("-topmost", True)
        tip.configure(fg_color=CARD_BG)
        
        # Position near cursor
        x = event.x_root + 15
        y = event.y_root + 15
        
        # Ensure it doesn't go off-screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        if x + 380 > screen_w:
            x = event.x_root - 395
        if y + 200 > screen_h:
            y = event.y_root - 215
        
        tip.geometry(f"+{x}+{y}")
        
        # Content frame with border effect
        frame = ctk.CTkFrame(tip, fg_color=CARD_BG, corner_radius=12, 
                            border_width=2, border_color=ACCENT_COLOR)
        frame.pack(padx=2, pady=2, fill="both", expand=True)
        
        label = ctk.CTkLabel(frame, text=text, font=("Inter", 13), 
                            text_color=TEXT_COLOR, wraplength=350, justify="left")
        label.pack(padx=15, pady=15)
        
        close_btn = ctk.CTkButton(frame, text="✕ Close", width=60, height=25,
                                  fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
                                  text_color=TEXT_COLOR, font=("Inter", 11),
                                  command=tip.destroy)
        close_btn.pack(pady=(0, 10))
        
        # Auto-close after 15 seconds
        tip.after(15000, lambda: tip.destroy() if tip.winfo_exists() else None)
        
        # Close on click anywhere outside
        tip.bind("<FocusOut>", lambda e: tip.destroy() if tip.winfo_exists() else None)

    def _image_send_to_qt(self):
        """Transfers the current page image to Quick Translate Image Mode."""
        if not self.all_page_data or self.current_page_idx >= len(self.all_page_data):
            return
            
        # SPRINT 82: HARDENED Index & Path logic
        curr_idx = int(self.current_page_idx)
        data = self.all_page_data[curr_idx]
        
        # 1. Path Resolution (Metadata vs. Cache Fallback)
        img_path = data.get('cover_image_path')
        if not img_path or not os.path.exists(img_path):
            img_path = os.path.join(self.cache_dir, f"img_{curr_idx}.jpg")
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                
                # 2. Sync Languages automatically
                # If we detected a specific source language for this page, set it in QT
                src_lang_code = data.get('translator_src', 'auto')
                # Find the 'Nice Name' for the UI dropdown
                nice_name = "Auto-Detect"
                for name, info in self.languages.items():
                    if info.get('trans') == src_lang_code:
                        nice_name = name
                        break
                
                self.qi_src_lang.set(nice_name)
                self.qt_src_lang.set(nice_name)
                
                # 3. Switch Tab & Update
                self.tab_view.set("Quick Translate")
                self.qt_tabs.set("Image Mode")
                self._update_qi_preview(img)
                self.status_label.configure(text=f"📤 Sent Page {curr_idx + 1} to Quick Translate.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to send image: {e}")
        else:
            messagebox.showinfo("Wait", f"Page {curr_idx + 1} doesn't have an extracted image yet. Please wait for processing.")

    def _image_save_as(self):
        """Standard save dialog for the current page image."""
        if not self.all_page_data or self.current_page_idx >= len(self.all_page_data):
            return
            
        page_idx = self.current_page_idx
        img_path = os.path.join(self.cache_dir, f"img_{page_idx}.jpg")
        
        if os.path.exists(img_path):
            out_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
                initialfile=f"page_{page_idx + 1}_visual.png"
            )
            if out_path:
                try:
                    from shutil import copy2
                    copy2(img_path, out_path)
                    messagebox.showinfo("Saved", f"Image saved to: {out_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")
        else:
            messagebox.showinfo("Wait", "No visual image available for this page.")

    def _image_clear_qt(self):
        """Resets the Quick Translate image state."""
        self.current_qi_img = None
        self.current_qi_path = None
        self.qi_preview_lbl.configure(image=None, text="Drag & Drop Image or Ctrl+V to Paste")
        self.qi_output.configure(state="normal")
        self.qi_output.delete("1.0", "end")
        self.qi_output.configure(state="disabled")
        
    def build_ui_skeleton(self):
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()

        self.main_container.grid_columnconfigure(0, weight=1) # Page#
        self.main_container.grid_columnconfigure(1, weight=5) # Transcription
        self.main_container.grid_columnconfigure(2, weight=5) # 1-1 OR Visual Feed
        self.main_container.grid_columnconfigure(3, weight=5) # English
        self.main_container.grid_rowconfigure(1, weight=1) # Allow text boxes to expand vertically
        
        # Column 0: Page Number
        self.page_number_label = ctk.CTkLabel(self.main_container, text="Page 0", font=("Inter", 13, "bold"), text_color=ACCENT_COLOR)
        self.page_number_label.grid(row=1, column=0, padx=10, pady=20, sticky="n")

        # Create Paper Sheet Containers for Column 1, 2, 3
        def create_paper_sheet(col):
            frame = ctk.CTkFrame(self.main_container, fg_color=PAPER_SHEET_BG, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
            frame.grid(row=1, column=col, padx=8, pady=15, sticky="nsew")
            
            # Use a slightly darker color for the textboxes to create depth
            box = ctk.CTkTextbox(frame, fg_color="transparent", text_color=TEXT_COLOR, 
                                 font=("Inter", 14), wrap="word", border_width=0, corner_radius=0)
            box.pack(padx=15, pady=15, fill="both", expand=True)
            return box, frame

        self.orig_text, self.orig_container = create_paper_sheet(1)
        self.lit_text, self.mid_container = create_paper_sheet(2)
        self.trans_text, self.trans_container = create_paper_sheet(3)
        
        # Bind key release events for all textboxes for auto-save
        self.orig_text.bind("<KeyRelease>", lambda e: self.schedule_save_edits())
        self.lit_text.bind("<KeyRelease>", lambda e: self.schedule_save_edits())
        self.trans_text.bind("<KeyRelease>", lambda e: self.schedule_save_edits())
        
        # SPRINT 39: Bind right-click "Reader Magic" context menu
        for box in [self.orig_text, self.lit_text, self.trans_text]:
            box.bind("<Button-3>", self.show_reader_context_menu)
        
        # Special Label for image pages in column 2 (Visual Feed)
        self.visual_img_label = ctk.CTkLabel(self.mid_container, text="", fg_color="transparent")
        # Hidden initially
        # SPRINT 39 Extension: Bind right-click Image Magic
        self.visual_img_label.bind("<Button-3>", lambda e: self.show_image_context_menu(e, "translation"))
        self.visual_img_label.pack_forget()
        
    def schedule_save_edits(self):
        """Sets a debounced timer so we don't save to disk on every single keystroke."""
        if self.save_timer is not None:
            self.after_cancel(self.save_timer)
        self.save_timer = self.after(1500, self.save_edits_to_cache)
        
    def on_closing(self):
        """Gracefully shuts down all background tasks before exiting."""
        self.stop_requested = True
        try:
            if hasattr(self, 'ocr_executor') and self.ocr_executor:
                self.ocr_executor.shutdown(wait=False)
            if hasattr(self, 'translation_executor') and self.translation_executor:
                self.translation_executor.shutdown(wait=False)
            if hasattr(self, 'active_executor') and self.active_executor:
                try: self.active_executor.shutdown(wait=False, cancel_futures=True)
                except: pass
            
            if self.supabase and hasattr(self.supabase.auth, 'close'):
                self.supabase.auth.close()
        except: pass
        self.destroy()
        sys.exit(0)

    def literal_word_by_word(self, trans, text):
        if not text: return ""
        words = text.split()
        lits = []
        for w in words[:200]: # Cap for speed
            try: lits.append(trans.translate(w))
            except: lits.append(w)
        return " ".join(lits)

    def process_single_page(self, doc, page_num, translator, ocr_lang_code, is_auto, selected_lang, is_rtl, single_page_mode, pages_to_process, num_pages_to_process):
        """Processes a single page sequentially using the ocr_worker for consistency."""
        session_at_start = self.current_session
        try:
            # Re-confirm page dimensions
            page_rect = doc[0].rect
            page_width = page_rect.width
            translator_src = translator.source
            
            # Update status
            progress_text = f"Processing Page {page_num + 1} of {num_pages_to_process}..."
            self.schedule_ui_update(lambda t=progress_text: self.status_label.configure(text=t))
            if num_pages_to_process > 0:
                self.schedule_ui_update(lambda v=(page_num + 1) / num_pages_to_process: self.progress_bar.set(v))
            
            # UX: Render the page immediately so the user sees something while OCR works (if user is looking at it)
            if self.current_session == session_at_start and page_num == self.current_page_idx:
                self.schedule_ui_update(lambda idx=page_num: self.render_page(int(idx)))
            
            # Phase 1: OCR Extraction (Direct call to worker for sequential safety)
            res = ocr_worker(page_num, self.current_pdf_path, ocr_lang_code, is_auto, 
                           self.languages, translator_src, page_width, is_rtl, self.tesseract_path)
            
            if res and isinstance(res, dict) and "error" not in res:
                print(f"[MainThread-Sequential] Received OCR result for Page {page_num}")
                p_num = int(res.get('page', page_num))
                if p_num < len(self.all_page_data):
                    if self.all_page_data[p_num] is None:
                        self.all_page_data[p_num] = res
                    else:
                        self.all_page_data[p_num].update(res)
                self.save_page_to_cache(res)
                
                # Re-render if session is still valid AND we are still on this page
                if self.current_session == session_at_start and p_num == self.current_page_idx:
                    self.schedule_ui_update(lambda idx=p_num: self.render_page(int(idx)))
                
                # Phase 2: Translation
                source_text = str(res.get('original', ''))
                src_lang = str(res.get('translator_src', 'auto'))
                
                if len(source_text) > 5 and "[No Text" not in source_text:
                    # Context Gathering (Previous and Next page text)
                    context_data = None
                    if self.selected_engine == "GPT-4o":
                        context_data = {
                            "prev_page": self.all_page_data[p_num-1].get('original', '') if p_num > 0 else "",
                            "next_page": self.all_page_data[p_num+1].get('original', '') if p_num < len(self.all_page_data)-1 else ""
                        }

                    t_res = translate_worker(
                        text=source_text, 
                        translator_src=src_lang, 
                        engine=self.selected_engine,
                        deepl_key=self.deepl_key, 
                        openai_key=self.openai_key, 
                        ollama_model=getattr(self, 'ollama_model', 'llama3'),
                        glossary=self.glossary, 
                        context=context_data
                    )
                    if "error" not in t_res and self.current_session == session_at_start:
                        self.all_page_data[p_num]['english'] = t_res['translated']
                        self.all_page_data[p_num]['literal'] = t_res['literal']
                        self.save_page_to_cache(self.all_page_data[p_num])
                        self.schedule_ui_update(lambda i=p_num: self.render_page(int(i)))
                    elif "error" in t_res and self.current_session == session_at_start:
                        self.all_page_data[p_num]['english'] = f"[Translation Failed: {str(t_res.get('error', 'Unknown'))[:100]}]"
                        self.save_page_to_cache(self.all_page_data[p_num])
                        self.schedule_ui_update(lambda i=p_num: self.render_page(int(i)))
                
                self.schedule_ui_update(lambda i=p_num + 1: self.status_label.configure(text=f"Finished Page {i}"))
            else:
                err_msg = res.get('error', 'Unknown Error') if res else 'Worker failed'
                print(f"[Sequential] OCR Error Page {page_num}: {err_msg}")
                err_data = {
                    "page": page_num,
                    "original": f"[Error on Page {page_num + 1}]",
                    "english": f"Processing failed: {err_msg}",
                    "literal": "",
                    "is_image": False,
                    "is_rtl_page": False,
                    "is_centered": False,
                    "is_cover": False
                }
                if page_num < len(self.all_page_data):
                    self.all_page_data[page_num] = err_data
                if self.current_session == session_at_start:
                    self.schedule_ui_update(lambda idx=page_num: self.render_page(int(idx)))
        except Exception as e:
            print(f"[Sequential] Page {page_num} failed: {e}")

    def preprocess_for_ocr(self, img):
        # Neural net OCR works better on high contrast, sharp images
        gray = ImageOps.grayscale(img)
        # Increase contrast and sharpen to help character recognition
        enhanced = ImageOps.autocontrast(gray, cutoff=2)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        return sharpened

    def deep_preprocess_for_ocr(self, pil_img):
        # 1. Convert PIL to OpenCV format (Grayscale)
        open_cv_image = np.array(pil_img.convert('L'))
        
        # 1.5 Upscale the image to boost clarity for tiny text before further processing
        # This acts as a magnifying glass, making tiny/fuzzy letters much clearer for Tesseract
        open_cv_image = cv2.resize(open_cv_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # 2. Border Detection & Removal (Rescue for decorative frames like Page 3)
        # We look for large contours that might be a decorative border
        _, thresh_border = cv2.threshold(open_cv_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = open_cv_image.shape
        img_area = h_img * w_img
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # If a contour covers > 60% of image with a large aspect ratio, it's likely a border
            if area > img_area * 0.6:
                x, y, w_border, h_border = cv2.boundingRect(cnt)
                # If it's near the edges, we white out the border area
                if x < 50 or y < 50 or (x + w_border) > w_img - 50:
                    # Create a mask to keep only the inside
                    mask = np.zeros_like(open_cv_image)
                    # Shrink the border rectangle slightly to keep text
                    inside_rect = (x + 30, y + 30, w_border - 60, h_border - 60)
                    cv2.rectangle(mask, (inside_rect[0], inside_rect[1]), 
                                  (inside_rect[0] + inside_rect[2], inside_rect[1] + inside_rect[3]), 255, -1)
                    # Apply mask: anything outside become white
                    open_cv_image = cv2.bitwise_and(open_cv_image, mask)
                    open_cv_image[mask == 0] = 255
                    break

        # 3. Bilateral Filtering (Preserves dots/diacritics while removing noise)
        denoised = cv2.bilateralFilter(open_cv_image, 9, 75, 75)
        
        # 3b. Stronger Contrast & Sharpening (Academic Tuning)
        # Convert back to PIL for high-quality sharpening
        temp_pil = Image.fromarray(denoised)
        # Apply high-pass sharpening to define thin characters (brackets, dots)
        temp_pil = temp_pil.filter(ImageFilter.SHARPEN)
        temp_pil = ImageOps.autocontrast(temp_pil, cutoff=1)
        # SPRINT 31: Auto-Contrast Equalization to help OCR find faint numbers
        temp_pil = ImageOps.equalize(temp_pil)
        # Back to CV
        denoised = np.array(temp_pil)

        # 4. Deskewing
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        else:
            angle = 0
            
        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # 5. Hybrid Thresholding (Otsu for styled fonts if Deep Cleanup is on)
        if self.deep_cleanup_var.get():
            # Otsu is better for thick stylized fonts on uniform backgrounds (Covers)
            _, final_thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptive is better for standard pages with uneven lighting
            final_thresh = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # 6. "Quiet Zone" Padding
        final_thresh = cv2.copyMakeBorder(final_thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255])
        
        # SPRINT 31: Speckle Filter XL (Increased threshold to 15px area)
        # Removes larger scanner noise/dust that causes "Ghosting"
        labels_info = cv2.connectedComponentsWithStats(cv2.bitwise_not(final_thresh), connectivity=8)
        num_labels, labels, stats, _ = labels_info
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 15:
                final_thresh[labels == i] = 255
                
        # SPRINT 30: Ink Density Calculation
        ink_pixels = np.sum(final_thresh == 0)
        ink_density = ink_pixels / final_thresh.size
        
        return Image.fromarray(final_thresh), ink_density

    def restore_arabic_numerals(self, text):
        # Maps Western (English) digits back to Eastern Arabic digits for UI aesthetics
        mapping = {'0':'٠','1':'١','2':'٢','3':'٣','4':'٤','5':'٥','6':'٦','7':'٧','8':'٨','9':'٩'}
        for w, e in mapping.items():
            text = text.replace(w, e)
        return text

    def normalize_to_western_digits(self, text):
        # Maps Eastern Arabic digits BACK to Western for the translation engine
        mapping = {'٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'}
        for e, w in mapping.items():
            text = text.replace(e, w)
        return text

    def detect_script(self, text):
        """Returns the likely Tesseract OCR code(s) joined by '+' and RTL status."""
        if not text or len(text) < 5:
            return {"ocr": "eng", "trans": "auto", "rtl": False}
            
        # SPRINT 81: "B.S. Filter" - Remove noise and symbols to get clean script count
        clean_text = re.sub(r'[^a-zA-Z\u0600-\u06FF\u0590-\u05FF\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\s\d]', '', text)
        
        # SPRINT 81: Granular script detection logic
        counts = {
            "ara": len(re.findall(r'[\u0600-\u06FF]', clean_text)),
            "heb": len(re.findall(r'[\u0590-\u05FF]', clean_text)),
            "ell": len(re.findall(r'[\u0370-\u03FF]', clean_text)),
            "rus": len(re.findall(r'[\u0400-\u04FF]', clean_text)),
            "chi_sim": len(re.findall(r'[\u4E00-\u9FFF]', clean_text)),
            "jpn": len(re.findall(r'[\u3040-\u30FF]', clean_text)),
            "kor": len(re.findall(r'[\uAC00-\uD7AF]', clean_text)),
            "latin": len(re.findall(r'[a-zA-Z\u00C0-\u017F]', clean_text))
        }
        
        # Mapping back to translator codes
        trans_map = {
            "ara": "ar", "heb": "iw", "ell": "el", "rus": "ru", "chi_sim": "zh-CN", 
            "jpn": "ja", "kor": "ko", "latin": "en"
        }
        
        total_chars = sum(counts.values())
        if total_chars < 5:
            return {"ocr": "eng", "trans": "auto", "rtl": False}
            
        detected_ocr_codes = []
        is_rtl = False
        
        # Hybrid Detection: Keep anything significant (> 10%)
        potential_codes = ["ara", "heb", "ell", "rus", "chi_sim", "jpn", "kor", "latin"]

        for code in potential_codes:
            char_count = counts.get(code, 0)
            if char_count > 2 or (char_count / total_chars > 0.1):
                if code == "latin":
                    detected_ocr_codes.extend(["eng", "fra", "spa", "deu", "ita"])
                else:
                    detected_ocr_codes.append(code)
                
                if code in ["ara", "heb"]:
                    is_rtl = True
        
        if not detected_ocr_codes:
            detected_ocr_codes = ["eng"]
            
        # Clean duplicates (e.g. from latin extension)
        final_ocr = "+".join(list(dict.fromkeys(detected_ocr_codes)))
        
        # Determine likely primary source language
        if len(list(dict.fromkeys(detected_ocr_codes))) > 5:
            final_trans = "auto"
        else:
            # Pick the strongest contributor
            final_trans = trans_map.get(max(counts, key=counts.get), "auto")
            
        return {
            "ocr": final_ocr,
            "trans": final_trans,
            "rtl": is_rtl
        }

    def robust_translate(self, translator, text):
        """Attempts to translate text with exponential backoff, length safety, and stealth diagnostics."""
        if not text: return ""
        
        # SPRINT 29: Normalize numerals to Western for Google accuracy
        text = self.normalize_to_western_digits(text)
        
        # SPRINT 31: Unicode Isolation Shield (LRI/PDI markers)
        text = re.sub(r'(\d+[\-/]\d+[\-/]?[0-9]*[\-/]?[0-9]*)', u' \u2066\\1\u2069 ', text) 
        
        text = self.rebalance_scripts(text)
        
        if len(text) < 3: return text
        
        # SPRINT 27: Tight Character Limit (2,500 chars) for maximum stealth
        if len(text) > 2500:
            parts = [text[i:i+2500] for i in range(0, len(text), 2500)]
            results = [self.robust_translate(translator, p) for p in parts]
            return "\n".join(results)

        import time
        import random
        # 3-tier backoff for total reliability
        for attempt in range(3):
            try:
                translated = "[Translation Error]"
                if self.selected_engine == "DeepL" and self.deepl_key:
                    import deepl
                    t = deepl.Translator(self.deepl_key)
                    src = None if translator.source == 'auto' else translator.source.upper()
                    res = t.translate_text(text, target_lang="EN-US", source_lang=src)
                    translated = res.text
                elif self.selected_engine == "GPT-4o" and self.openai_key:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_key)
                    
                    gloss_instr = ""
                    if self.glossary:
                        gloss_list = [f"'{k}' -> '{v}'" for k, v in self.glossary.items()]
                        gloss_instr = "IMPORTANT: Use the following custom terminology glossary: " + ", ".join(gloss_list) + "."
                        
                    response = client.chat.completions.create(
                      model="gpt-4o",
                      messages=[
                        {"role": "system", "content": f"You are a professional translator. Translate the following text from {translator.source} to English. Preserve formatting and nuances. Output ONLY the translated text. {gloss_instr}"},
                        {"role": "user", "content": text}
                      ]
                    )
                    translated = response.choices[0].message.content.strip()
                else: # Default Google
                    # Pre-processing Glossary for Google
                    if self.glossary:
                        for orig, target in self.glossary.items():
                            pattern = re.compile(re.escape(orig), re.IGNORECASE)
                            text = pattern.sub(target, text)
                            
                    fresh_translator = GoogleTranslator(source=translator.source, target=translator.target)
                    translated = str(fresh_translator.translate(text))
                    
                    # Passthrough check
                    if translated.strip().lower() == text.strip().lower() and len(text) > 15:
                        auto_translator = GoogleTranslator(source='auto', target='en')
                        translated = str(auto_translator.translate(text))
                
                return translated
            except Exception as e:
                print(f"DEBUG: Translation Attempt {attempt+1} failed ({len(text)} chars): {str(e)}")
                time.sleep(2 ** attempt + random.uniform(0.1, 0.5))
                
        return "[Translation Failed]"

    def schedule_ui_update(self, func):
        self.after(0, func)

if __name__ == "__main__":
    import multiprocessing
    # CRITICAL: Freeze support MUST be before any multiprocessing calls
    multiprocessing.freeze_support()
    
    # SPRINT 43: Explicitly set spawn method for Windows stability
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except:
        pass
        
    app = PDFTranslatorApp()
    app.mainloop()
