"""
ui_components.py - Settings, menus, quick translate, and UI utility methods.
Extracted from main.py during The Great Refactor as a mixin class.
"""
import os
import json
import time
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import requests

from utils import (
    APP_VERSION, humanize_error, resource_path, get_app_path,
    PAPER_BG, TEXT_COLOR, TEXT_DIM, HEADER_BG, ACCENT_COLOR, ACCENT_HOVER,
    PAPER_SHEET_BG, BORDER_COLOR, CARD_BG, FRAME_BG,
    BTN_PRIMARY, BTN_PRIMARY_HOVER, BTN_SECONDARY, BTN_SECONDARY_HOVER,
    INPUT_BG, BTN_DANGER, BTN_DANGER_HOVER, BTN_SUCCESS, BTN_SUCCESS_HOVER,
    BTN_WARNING, BTN_WARNING_HOVER,
)
from workers import translate_worker
try:
    import windnd
except ImportError:
    windnd = None



class UIComponentsMixin:
    """Mixin class providing settings, menus, quick translate, and UI utilities."""

    def check_for_updates(self):
        """Checks Supabase Storage for a newer manifest.json."""
        if not self.supabase: return
        try:
            # We assume manifest.json is in a public bucket 'app-releases'
            import requests # We might need to import requests if not already there
            # Since we are using supabase client, we can try to get public url
            res = self.supabase.storage.from_("app-releases").download("manifest.json")
            if res:
                manifest = json.loads(res.decode('utf-8'))
                remote_version = manifest.get("version", "1.0.0")
                if remote_version > APP_VERSION:
                    update_url = manifest.get("url", "")
                    def notify():
                        msg = f"A new version ({remote_version}) is available!\n\n"
                        if update_url:
                            msg += f"Download here: {update_url}\n\n"
                        msg += "Would you like to visit the download page?"
                        if messagebox.askyesno("Update Available", msg):
                             import webbrowser
                             webbrowser.open(update_url or "https://github.com/PDF-Translator-Pro")
                    self.schedule_ui_update(notify)
        except Exception as e:
            print(f"Update Check Fail: {e}")

    def show_about(self):
        messagebox.showinfo("About", f"Bilingual Book Translator Pro\nVersion {APP_VERSION}\n\nA professional tool for PDF translation and dual-language book creation.\nCreated with Antigravity.")

    def toggle_settings_popdown(self):
        """Toggles the visibility of the settings popdown frame."""
        if self.settings_popdown and self.settings_popdown.winfo_exists():
            self.settings_popdown.destroy()
            self.settings_popdown = None
            return
            
        # Create Popdown Frame
        self.settings_popdown = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER_COLOR, width=420)
        self.settings_popdown.place(relx=0.96, rely=0.07, anchor="ne")
        self.settings_popdown.lift()
        
        title_lbl = ctk.CTkLabel(self.settings_popdown, text="Translation Options", font=("Inter", 14, "bold"), text_color=TEXT_COLOR)
        title_lbl.pack(pady=(15, 5))
        
        info_var = ctk.StringVar(value="Hover over an option for details.")
        
        def on_enter(text):
            info_var.set(text)
            
        def on_leave(e):
            info_var.set("Hover over an option for details.")
            
        # 1-1 Pro Mode
        cb_one_one = ctk.CTkCheckBox(self.settings_popdown, text="1-1 Pro Mode (Bilingual Layout)", variable=self.one_one_var, command=self.toggle_one_one, text_color=TEXT_COLOR)
        cb_one_one.pack(anchor="w", padx=20, pady=10)
        cb_one_one.bind("<Enter>", lambda e: on_enter("Creates a strict side-by-side translation.\nOriginal left, translated right."))
        cb_one_one.bind("<Leave>", on_leave)

        # Deep Cleanup
        cb_deep = ctk.CTkCheckBox(self.settings_popdown, text="Deep Cleanup (Clean Mode)", variable=self.deep_cleanup_var, text_color=TEXT_COLOR)
        cb_deep.pack(anchor="w", padx=20, pady=10)
        cb_deep.bind("<Enter>", lambda e: on_enter("Removes headers/watermarks for cleaner output."))
        cb_deep.bind("<Leave>", on_leave)
        
        # Fast Mode
        cb_fast = ctk.CTkCheckBox(self.settings_popdown, text="Fast Mode (Low Quality OCR)", variable=self.speed_var, text_color=TEXT_COLOR)
        cb_fast.pack(anchor="w", padx=20, pady=10)
        cb_fast.bind("<Enter>", lambda e: on_enter("Speeds up processing by sacrificing layout retention."))
        cb_fast.bind("<Leave>", on_leave)
        
        # Eco Mode
        cb_eco = ctk.CTkCheckBox(self.settings_popdown, text="Eco Mode (Save Resources)", variable=self.eco_var, text_color=TEXT_COLOR)
        cb_eco.pack(anchor="w", padx=20, pady=10)
        cb_eco.bind("<Enter>", lambda e: on_enter("Reduces thread speed/memory for low-end devices."))
        cb_eco.bind("<Leave>", on_leave)

        # SPRINT 39: AI Engine Selection
        ctk.CTkLabel(self.settings_popdown, text="AI Translation Engine:", font=("Inter", 12, "bold")).pack(anchor="w", padx=20, pady=(10, 0))
        
        def on_engine_change(choice):
            self.selected_engine = choice
            self.save_settings()
            
        engine_menu = ctk.CTkOptionMenu(self.settings_popdown, values=["Google", "DeepL", "GPT-4o", "Ollama (Local)"], command=on_engine_change)
        engine_menu.set(self.selected_engine)
        engine_menu.pack(fill="x", padx=20, pady=5)
        engine_menu.bind("<Enter>", lambda e: on_enter("Choose your translation provider.\nGoogle is free, DeepL/GPT-4o require keys, Ollama is local/free."))
        engine_menu.bind("<Leave>", on_leave)
        
        # DeepL Key Entry
        if not hasattr(self, 'deepl_var'):
            self.deepl_var = ctk.StringVar(value=self.deepl_key)
            self.deepl_var.trace_add("write", lambda *a: (setattr(self, 'deepl_key', self.deepl_var.get().strip()), self.save_settings()))
            
        deepl_entry = ctk.CTkEntry(self.settings_popdown, textvariable=self.deepl_var, placeholder_text="DeepL API Key...", height=30)
        deepl_entry.pack(fill="x", padx=20, pady=5)
        deepl_entry.bind("<Enter>", lambda e: on_enter("Enter your DeepL API key (Free or Pro)."))
        deepl_entry.bind("<Leave>", on_leave)
        
        # OpenAI Key Entry
        if not hasattr(self, 'openai_var'):
            self.openai_var = ctk.StringVar(value=self.openai_key)
            self.openai_var.trace_add("write", lambda *a: (setattr(self, 'openai_key', self.openai_var.get().strip()), self.save_settings()))
            
        openai_entry = ctk.CTkEntry(self.settings_popdown, textvariable=self.openai_var, placeholder_text="OpenAI API Key...", height=30)
        openai_entry.pack(fill="x", padx=20, pady=5)
        openai_entry.bind("<Enter>", lambda e: on_enter("Enter your OpenAI API key for GPT-4o."))
        openai_entry.bind("<Leave>", on_leave)

        # Ollama Model Entry
        if not hasattr(self, 'ollama_var'):
            self.ollama_var = ctk.StringVar(value=getattr(self, 'ollama_model', 'llama3'))
            self.ollama_var.trace_add("write", lambda *a: (setattr(self, 'ollama_model', self.ollama_var.get().strip()), self.save_settings()))
            
        ollama_entry = ctk.CTkEntry(self.settings_popdown, textvariable=self.ollama_var, placeholder_text="Ollama Model (e.g. llama3)", height=30)
        ollama_entry.pack(fill="x", padx=20, pady=(5, 15))
        ollama_entry.bind("<Enter>", lambda e: on_enter("Enter your installed Ollama model (e.g. llama3, mistral)."))
        ollama_entry.bind("<Leave>", on_leave)

        # Glossary Manager Button
        ctk.CTkLabel(self.settings_popdown, text="Custom Terminology:", font=("Inter", 12, "bold")).pack(anchor="w", padx=20, pady=(5, 0))
        gloss_btn = ctk.CTkButton(self.settings_popdown, text="📖 Open Glossary Manager", 
                                 fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER, text_color="#FFFFFF",
                                 command=self.open_glossary_manager)
        gloss_btn.pack(fill="x", padx=20, pady=(5, 10))
        gloss_btn.bind("<Enter>", lambda e: on_enter("Set custom translations for names, places,\nor technical terms to keep them consistent."))
        gloss_btn.bind("<Leave>", on_leave)

        info_frame = ctk.CTkFrame(self.settings_popdown, fg_color=INPUT_BG, height=40, corner_radius=5)
        info_frame.pack(fill="x", padx=15, pady=(10, 15))
        info_frame.pack_propagate(False)

        info_lbl = ctk.CTkLabel(info_frame, textvariable=info_var, font=("Inter", 11, "italic"), text_color=TEXT_DIM)
        info_lbl.pack(expand=True)

    def create_menu_bar(self):
        """Create a professional Windows-style top menu bar."""
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        # 1. File Menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="↗ Upload PDF...", command=self.upload_pdf, accelerator="Ctrl+O")
        self.file_menu.add_command(label="↺ Resume Last Session", command=self.resume_session)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="📚 Open Global Library", command=lambda: self.tab_view.set("Global Library"))
        self.file_menu.add_command(label="📥 Inbox (Requests)", command=self.show_inbox)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_closing)

        # 2. View Menu (Toggles)
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Toggle Theme (Light/Dark)", command=self.toggle_theme)
        self.view_menu.add_separator()
        self.view_menu.add_command(label="⚙ Preferences / Fonts", command=self.open_settings)

        # 3. Export Menu
        self.export_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Export", menu=self.export_menu)
        self.export_menu.add_command(label="Side-by-Side PDF", command=lambda: self.handle_export("Export PDF"))
        self.export_menu.add_command(label="Microsoft Word (.docx)", command=lambda: self.handle_export("Export Word"))
        self.export_menu.add_command(label="eBook (ePub)", command=lambda: self.handle_export("Export ePub"))

        # 4. Help Menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Documentation", command=lambda: messagebox.showinfo("Help", "Visit documentation online for more details."))
        self.help_menu.add_separator()
        self.help_menu.add_command(label="About", command=self.show_about)

        self.bind_all("<Control-o>", lambda e: self.upload_pdf())

        # --- SPRINT 98: Pro Keyboard Shortcuts ---
        def _check_typing(e):
            import tkinter as tk
            import customtkinter as ctk
            win = self.focus_get()
            if not win:
                return False
            # Check if the active widget is a text input
            return isinstance(win, (tk.Entry, tk.Text, ctk.CTkEntry, ctk.CTkTextbox))
            
        def _force_save(e):
            self.save_edits_to_cache()
            self.status_label.configure(text="💾 Edits Saved.")

        self.bind_all("<Control-s>", _force_save)
        self.bind_all("<Control-r>", lambda e: self.rescan_current_page())
        self.bind_all("<Control-e>", lambda e: self.handle_export("Export PDF"))
        self.bind_all("<Control-b>", lambda e: self.toggle_bookmark())
        self.bind_all("<Control-d>", lambda e: self.toggle_theme())
        self.bind_all("<Control-f>", lambda e: self.search_entry.focus() if hasattr(self, 'search_entry') else None)
        self.bind_all("<Control-j>", lambda e: self.jump_entry.focus() if hasattr(self, 'jump_entry') else None)
        self.bind_all("<Control-=>", lambda e: self.update_font_size(min(self.font_size + 2, 30)))
        self.bind_all("<Control-minus>", lambda e: self.update_font_size(max(self.font_size - 2, 10)))
        self.bind_all("<Escape>", lambda e: self.focus_set())
        
        # Smart Navigation (Focus-Aware)
        self.bind_all("<Right>", lambda e: self.change_page("next") if not _check_typing(e) else None)
        self.bind_all("<Left>", lambda e: self.change_page("prev") if not _check_typing(e) else None)
        # ----------------------------------------

        # --- SPRINT 98: Drag-and-Drop Setup ---
        if windnd:
            try:
                # SPRINT 101: ENABLE UNICODE SUPPORT
                # We set force_unicode=True so windnd uses DragQueryFileW (Wide/Unicode).
                windnd.hook_dropfiles(self, self.on_file_drop, force_unicode=True)
                
                # Admin Check (UIPI Warning)
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                admin_status = "[ADMIN]" if is_admin else "[USER]"
                print(f"[UI] Drag-and-Drop {admin_status} Unicode listener hooked to ID: {self.winfo_id()}.")
            except Exception as e:
                print(f"[UI] Drag-and-Drop hook failed: {e}")
        else:
            print("[UI] Drag-and-Drop library (windnd) NOT FOUND. Feature disabled.")
        # --------------------------------------

        self.is_paused = False
        self.stop_requested = False
        self.current_pdf_path = None
        self.bookmarks = [] # List of page indices
        self.current_theme = "Dark" # State tracker for theme toggle
        self.is_admin = False # SPRINT 34 Admin functionality

        # Storage and Pagination
        self.all_page_data = [] # List of {page, original, english, literal, ...}
        self.current_page_idx = 0
        self.total_pages = 0
        self.cache_dir = get_app_path(".translator_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Cloud Sync Queue (prevents RAM spikes from spawning 100s of threads)
        self.sync_queue = []
        self.sync_worker_active = False
        self.sync_lock = threading.Lock()
            
        # LRU Image Cache (max 5 images to prevent RAM crashes)
        self.image_cache = {}
            
        # UI Recycling References
        self.page_number_label = ctk.CTkLabel(self, text="")
        self.orig_text = ctk.CTkTextbox(self)
        self.lit_text = ctk.CTkTextbox(self)
        self.visual_img_label = ctk.CTkLabel(self, text="")
        self.trans_text = ctk.CTkTextbox(self)
        
        # SPRINT 33: Workspace Tabs
        self.tab_view = ctk.CTkTabview(self, fg_color=PAPER_BG)
        self.tab_view.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=10, pady=5)
        
        self.tab_pdf = self.tab_view.add("PDF Translator")
        self.tab_quick = self.tab_view.add("Quick Translate")
        self.tab_library = self.tab_view.add("Global Library")
        
        # Configure PDF Tab layout
        self.tab_pdf.grid_columnconfigure(0, weight=1)
        self.tab_pdf.grid_rowconfigure(1, weight=1)

        # Initialize Library UI
        self.build_library_ui()
        
        self.orig_container = ctk.CTkFrame(self.tab_pdf)
        self.mid_container = ctk.CTkFrame(self.tab_pdf)
        self.trans_container = ctk.CTkFrame(self.tab_pdf)

        # Headers Frame (Sticky) inside PDF Tab
        self.header_frame = ctk.CTkFrame(self.tab_pdf, fg_color=PAPER_BG, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20)
        
        # Main Container (PDF Tab)
        self.main_container = ctk.CTkFrame(self.tab_pdf, fg_color=PAPER_BG, corner_radius=0)
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        
        # Debounce timer for text edits
        self.save_timer = None
        
        # Build the permanent UI skeleton once
        self.build_ui_skeleton()
        self.build_quick_translate_ui()
        
        # Footer Frame for Navigation inside PDF Tab
        self.footer_frame = ctk.CTkFrame(self.tab_pdf, fg_color=HEADER_BG, height=50, corner_radius=0)
        self.footer_frame.grid(row=2, column=0, sticky="ew")
        self.tab_pdf.grid_rowconfigure(2, weight=0)
        
        self.nav_first = ctk.CTkButton(self.footer_frame, text="⇤ First", width=80, fg_color=BTN_SECONDARY, text_color=TEXT_COLOR, hover_color=BTN_SECONDARY_HOVER, command=lambda: self.change_page("first"))
        self.nav_first.pack(side="left", padx=10, pady=10)
        
        self.nav_prev = ctk.CTkButton(self.footer_frame, text="← Prev", width=80, fg_color=BTN_SECONDARY, text_color=TEXT_COLOR, hover_color=BTN_SECONDARY_HOVER, command=lambda: self.change_page("prev"))
        self.nav_prev.pack(side="left", padx=5, pady=10)
        
        self.page_label = ctk.CTkLabel(self.footer_frame, text="Page 0 of 0", text_color=TEXT_COLOR, font=("Inter", 12, "bold"))
        self.page_label.pack(side="left", padx=20, pady=10)
        
        self.nav_next = ctk.CTkButton(self.footer_frame, text="Next →", width=80, fg_color=BTN_SECONDARY, text_color=TEXT_COLOR, hover_color=BTN_SECONDARY_HOVER, command=lambda: self.change_page("next"))
        self.nav_next.pack(side="left", padx=5, pady=10)
        
        self.nav_last = ctk.CTkButton(self.footer_frame, text="Last ⇥", width=80, fg_color=BTN_SECONDARY, text_color=TEXT_COLOR, hover_color=BTN_SECONDARY_HOVER, command=lambda: self.change_page("last"))
        self.nav_last.pack(side="left", padx=10, pady=10)

        self.jump_label = ctk.CTkLabel(self.footer_frame, text="Jump:", text_color=TEXT_DIM)
        self.jump_label.pack(side="left", padx=(50, 5), pady=10)
        
        self.jump_entry = ctk.CTkEntry(self.footer_frame, width=50, fg_color=INPUT_BG, border_color=BORDER_COLOR, text_color=TEXT_COLOR)
        self.jump_entry.pack(side="left", padx=5, pady=10)
        self.jump_entry.bind("<Return>", lambda e: self.change_page("jump"))

        self.rescan_err_btn = ctk.CTkButton(self.footer_frame, text="⚠ Fix Errors", width=100, 
                                            fg_color=BTN_DANGER, hover_color=BTN_DANGER_HOVER, text_color="#FFFFFF", command=self.rescan_all_errors)
        self.rescan_err_btn.pack(side="right", padx=10, pady=10)

        self.rescan_btn = ctk.CTkButton(self.footer_frame, text="↻ Rescan Page", width=100, 
                                        fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR, command=self.rescan_current_page)
        self.rescan_btn.pack(side="right", padx=10, pady=10)

        self.bookmark_btn = ctk.CTkButton(self.footer_frame, text="★ Bookmark", width=100,
                                         fg_color=CARD_BG, text_color=ACCENT_COLOR, hover_color=BTN_SECONDARY_HOVER,
                                         command=self.toggle_bookmark)
        self.bookmark_btn.pack(side="right", padx=10, pady=10)

        self.update_column_config()
        self.setup_headers()
        self.row_counter = 1
        
        # Check for existing session at startup
        self.check_existing_session()
        
        # Trigger Login UI if Supabase is connected but no user
        if self.supabase and not self.current_user:
            self.after(500, self.update_avatar_ui)

    def toggle_theme(self):
        """Switches the application between Light and Dark mode."""
        if self.current_theme == "Dark":
            ctk.set_appearance_mode("Light")
            self.current_theme = "Light"
        else:
            ctk.set_appearance_mode("Dark")
            self.current_theme = "Dark"

    def load_settings(self):
        """Loads app settings and migrates glossary to terminology.json if needed."""
        self.font_family = "Times New Roman" # Default values
        self.font_size = 15
        self.bookmarks = []
        self.selected_engine = "Google"
        self.deepl_key = ""
        self.openai_key = ""
        self.tesseract_path = resource_path(os.path.join("bin", "tesseract", "tesseract.exe"))
        self.last_cache_dir = ""
        self.last_original_path = ""
        self.last_book_id = ""
        self.ollama_model = "llama3"

        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    s = json.load(f)
                    # Load core settings
                    self.font_family = s.get("font_family", "Times New Roman")
                    self.font_size = s.get("font_size", 15)
                    self.bookmarks = s.get("bookmarks", [])
                    self.deepl_key = s.get('deepl_key', '')
                    self.openai_key = s.get('openai_key', '')
                    self.tesseract_path = s.get('tesseract_path', resource_path(os.path.join("bin", "tesseract", "tesseract.exe")))
                    self.selected_engine = s.get('selected_engine', 'Google')
                    self.ollama_model = s.get('ollama_model', 'llama3')
                    self.last_cache_dir = s.get("last_cache_dir", "")
                    self.last_original_path = s.get("last_original_path", "")
                    self.last_book_id = s.get("last_book_id", "")
                    
                    # Migration Logic: If glossary exists in settings, move it
                    if 'glossary' in s and not os.path.exists(self.terminology_path):
                        self.glossary = s.get('glossary', {})
                        self.save_terminology()
                        # Clean up settings
                        if 'glossary' in s: del s['glossary']
                        with open(self.settings_path, 'w') as f2:
                            json.dump(s, f2)
            except: pass
            
        # Load Terminology
        if os.path.exists(self.terminology_path):
            try:
                with open(self.terminology_path, 'r') as f:
                    self.glossary = json.load(f)
            except: pass

    def save_settings(self):
        # SPRINT 79: Sync trackers with current session data before saving
        if getattr(self, "cache_dir", None):
            self.last_cache_dir = self.cache_dir
        if getattr(self, "current_pdf_path", None):
            self.last_original_path = self.current_pdf_path
        if getattr(self, "current_book_id", None):
            self.last_book_id = self.current_book_id

        s = {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "bookmarks": self.bookmarks,
            'deepl_key': self.deepl_key,
            'openai_key': self.openai_key,
            'tesseract_path': self.tesseract_path,
            'selected_engine': self.selected_engine,
            'ollama_model': getattr(self, 'ollama_model', 'llama3'),
            "last_cache_dir": self.last_cache_dir,
            "last_original_path": self.last_original_path,
            "last_book_id": self.last_book_id
        }
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(s, f)
            self.save_terminology()
        except: pass
            
    def search_text(self, event=None):
        query = self.search_var.get().lower()
        if not query or not self.all_page_data: return
        
        # Start searching from the next page
        start_idx = (self.current_page_idx + 1) % len(self.all_page_data)
        for i in range(len(self.all_page_data)):
            idx = (start_idx + i) % len(self.all_page_data)
            data = self.all_page_data[idx]
            text_pool = (str(data.get('original', '')) + 
                         str(data.get('english', '')) + 
                         str(data.get('literal', ''))).lower()
            if query in text_pool:
                self.current_page_idx = idx
                self.render_page(idx)
                self.status_label.configure(text=f"Found '{query}' on Page {idx+1}")
                return
        
        messagebox.showinfo("Search", f"No more matches found for '{query}'.")

    def open_glossary_manager(self):
        """Opens a window to manage custom terminology."""
        gloss_win = ctk.CTkToplevel(self)
        gloss_win.title("Terminology Glossary Manager")
        gloss_win.geometry("500x600")
        gloss_win.attributes("-topmost", True)
        gloss_win.configure(fg_color=CARD_BG)
        
        ctk.CTkLabel(gloss_win, text="Custom Glossary", font=("Inter", 18, "bold"), text_color=TEXT_COLOR).pack(pady=20)
        ctk.CTkLabel(gloss_win, text="Define rules like: 'Geralt' -> 'Geralt of Rivia'\nEnsures consistent translation across all engines.", 
                    font=("Inter", 12), text_color=TEXT_DIM).pack(pady=(0, 20))
        
        # List Frame
        list_frame = ctk.CTkScrollableFrame(gloss_win, fg_color=INPUT_BG, height=350)
        list_frame.pack(fill="both", padx=20, expand=True)
        
        def refresh_list():
            for widget in list_frame.winfo_children():
                widget.destroy()
            
            for orig, target in self.glossary.items():
                row = ctk.CTkFrame(list_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=f"{orig} → {target}", font=("Arial", 13)).pack(side="left", padx=10)
                
                def _del(o=orig):
                    self.glossary.pop(o, None)
                    self.save_settings()
                    refresh_list()
                    
                ctk.CTkButton(row, text="✕", width=30, height=25, fg_color="#E53935", 
                             command=_del).pack(side="right", padx=5)

        # Add Frame
        add_frame = ctk.CTkFrame(gloss_win, fg_color="transparent")
        add_frame.pack(fill="x", padx=20, pady=20)
        
        orig_entry = ctk.CTkEntry(add_frame, placeholder_text="Original Word/Name...", width=160)
        orig_entry.pack(side="left", padx=5)
        
        target_entry = ctk.CTkEntry(add_frame, placeholder_text="Target Translation...", width=160)
        target_entry.pack(side="left", padx=5)
        
        def _add():
            o = orig_entry.get().strip()
            t = target_entry.get().strip()
            if o and t:
                self.glossary[o] = t
                self.save_settings()
                orig_entry.delete(0, 'end')
                target_entry.delete(0, 'end')
                refresh_list()
            else:
                messagebox.showwarning("Input Error", "Please provide both words.")
                
        ctk.CTkButton(add_frame, text="+ Add", width=80, fg_color=ACCENT_COLOR, 
                     command=_add).pack(side="left", padx=5)
        
        refresh_list()

    def toggle_bookmark(self):
        if self.current_page_idx in self.bookmarks:
            self.bookmarks.remove(self.current_page_idx)
            self.status_label.configure(text=f"Removed Bookmark for Page {self.current_page_idx + 1}")
        else:
            self.bookmarks.append(self.current_page_idx)
            self.bookmarks.sort()
            self.status_label.configure(text=f"Bookmarked Page {self.current_page_idx + 1}")
        self.update_bookmark_ui()
        self.save_settings()

    def update_bookmark_ui(self):
        if self.current_page_idx in self.bookmarks:
            self.bookmark_btn.configure(text="★ Bookmarked", fg_color="#f39c12")
        else:
            self.bookmark_btn.configure(text="☆ Bookmark", fg_color="#f1c40f")

    def build_quick_translate_ui(self):
        """SPRINT 38: Builds a sub-tabbed layout for Text and Image translation."""
        self.tab_quick.grid_columnconfigure(0, weight=1)
        self.tab_quick.grid_rowconfigure(0, weight=1)
        
        # --- Internal TabView for Modes ---
        self.qt_tabs = ctk.CTkTabview(self.tab_quick, segmented_button_fg_color=HEADER_BG, 
                                      segmented_button_selected_color=ACCENT_COLOR)
        self.qt_tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        tab_text = self.qt_tabs.add("Text Mode")
        tab_image = self.qt_tabs.add("Image Mode")
        
        # === TEXT MODE UI ===
        tab_text.grid_columnconfigure(0, weight=1)
        tab_text.grid_columnconfigure(1, weight=1)
        tab_text.grid_rowconfigure(1, weight=1)

        # Top Bar for Text Mode
        t_top = ctk.CTkFrame(tab_text, fg_color="transparent")
        t_top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        all_langs = ["Auto-Detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean", "Arabic", "Hebrew", "Greek", "Turkish", "Dutch", "Polish", "Yiddish", "Latin"]
        
        ctk.CTkLabel(t_top, text="From:", font=("Inter", 12, "bold")).pack(side="left", padx=5)
        self.qt_src_lang = ctk.CTkComboBox(t_top, values=all_langs, width=130)
        self.qt_src_lang.set("Auto-Detect")
        self.qt_src_lang.pack(side="left", padx=5)
        
        ctk.CTkLabel(t_top, text="To:", font=("Inter", 12, "bold")).pack(side="left", padx=(15, 5))
        self.qt_tgt_lang = ctk.CTkComboBox(t_top, values=all_langs[1:], width=130) # No auto-detect for target
        self.qt_tgt_lang.set("English")
        self.qt_tgt_lang.pack(side="left", padx=5)

        # Input Box
        self.qt_input = ctk.CTkTextbox(tab_text, font=(self.font_family, self.font_size))
        self.qt_input.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.qt_input.insert("1.0", "Paste or type text here and press Ctrl+Enter...")
        self.qt_input.bind("<Control-Return>", lambda e: self.run_quick_translate())
        
        # SPRINT 83: Instant Stop - Bound to key release to detect deletions
        def _check_cancel(e):
            # SPRINT 93: Check either text input or OCR transcript box depending on mode
            txt_t = self.qt_input.get("1.0", "end-1c").strip()
            txt_i = self.qi_transcript.get("1.0", "end-1c").strip() if hasattr(self, 'qi_transcript') else ""
            
            # Check if the currently active box is cleared
            is_cleared = False
            if e.widget == self.qt_input._textbox: # Text Mode
                 if not txt_t or txt_t == "Paste or type text here and press Ctrl+Enter...":
                     is_cleared = True
            elif hasattr(self, 'qi_transcript') and e.widget == self.qi_transcript._textbox: # Image Mode
                 if not txt_i:
                     is_cleared = True

            if is_cleared:
                self.qt_cancel_flag = True
                if self.qt_btn.cget("text") == "Translating..." or self.qi_translate_btn.cget("text") == "Translating...":
                    self._finalize_qt("[Translation Stopped]")
                    if hasattr(self, 'qi_translation'):
                        self.qi_translation.configure(state="normal")
                        self.qi_translation.delete("1.0", "end")
                        self.qi_translation.insert("1.0", "[Translation Stopped]")
                        self.qi_translation.configure(state="disabled")
                        self.qi_translate_btn.configure(state="normal", text="Translate Image")

        self.qt_input.bind("<KeyRelease>", _check_cancel)
        self.qt_input.bind("<Control-Delete>", _check_cancel)
        self.qt_input.bind("<Control-BackSpace>", _check_cancel)
        
        # SPRINT 93: Image Mode bindings moved after initialization to avoid AttributeError
        
        # Output Box
        self.qt_output = ctk.CTkTextbox(tab_text, font=(self.font_family, self.font_size), fg_color=INPUT_BG)
        self.qt_output.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.qt_output.configure(state="disabled")

        self.qt_btn = ctk.CTkButton(tab_text, text="Translate Text", font=("Inter", 14, "bold"), 
                                    height=35, fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER,
                                    command=self.run_quick_translate)
        self.qt_btn.grid(row=2, column=0, columnspan=2, pady=15)

        # === IMAGE MODE UI ===
        # === IMAGE MODE UI ===
        tab_image.grid_columnconfigure(0, weight=1) # Preview (Smaller)
        tab_image.grid_columnconfigure(1, weight=4) # Results (Larger)
        tab_image.grid_rowconfigure(1, weight=1)

        # Top Bar for Image Mode
        i_top = ctk.CTkFrame(tab_image, fg_color="transparent")
        i_top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.qi_src_lang = ctk.CTkComboBox(i_top, values=all_langs, width=130)
        self.qi_src_lang.set("Auto-Detect")
        self.qi_src_lang.pack(side="left", padx=5)
        
        ctk.CTkLabel(i_top, text="➜", font=("Inter", 16)).pack(side="left", padx=5)
        
        self.qi_tgt_lang = ctk.CTkComboBox(i_top, values=all_langs[1:], width=130) # No auto-detect for target
        self.qi_tgt_lang.set("English")
        self.qi_tgt_lang.pack(side="left", padx=5)

        # Drop Zone / Preview
        self.qi_drop_zone = ctk.CTkFrame(tab_image, fg_color=FRAME_BG, border_width=2, border_color=ACCENT_COLOR)
        self.qi_drop_zone.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        self.qi_preview_lbl = ctk.CTkLabel(self.qi_drop_zone, text="Drag & Drop Image or Ctrl+V to Paste", 
                                           text_color=TEXT_DIM, font=("Inter", 14, "italic"))
        self.qi_preview_lbl.place(relx=0.5, rely=0.5, anchor="center")
        
        # Result Area: Split into Transcription and Translation (Side-by-Side)
        qi_results_frame = ctk.CTkFrame(tab_image, fg_color="transparent")
        qi_results_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        qi_results_frame.grid_rowconfigure(1, weight=1)
        qi_results_frame.grid_columnconfigure(0, weight=1)
        qi_results_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(qi_results_frame, text="Transcription (OCR):", font=("Inter", 12, "bold"), text_color=ACCENT_COLOR).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.qi_transcript = ctk.CTkTextbox(qi_results_frame, font=(self.font_family, self.font_size))
        self.qi_transcript.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        # SPRINT 88: Should be editable by default
        self.qi_transcript.configure(state="normal")
        
        # SPRINT 93: Bind Image Mode Instant Stop (moved here to avoid AttributeError)
        self.qi_transcript.bind("<KeyRelease>", _check_cancel)
        self.qi_transcript.bind("<Control-Delete>", _check_cancel)
        self.qi_transcript.bind("<Control-BackSpace>", _check_cancel)

        ctk.CTkLabel(qi_results_frame, text="Translation:", font=("Inter", 12, "bold"), text_color=BTN_SUCCESS).grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.qi_translation = ctk.CTkTextbox(qi_results_frame, font=(self.font_family, self.font_size), fg_color=INPUT_BG)
        self.qi_translation.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.qi_translation.configure(state="disabled")

        # Buttons
        i_btns = ctk.CTkFrame(tab_image, fg_color="transparent")
        i_btns.grid(row=2, column=0, columnspan=2, pady=15)

        ctk.CTkButton(i_btns, text="📁 Upload", width=100, command=self.upload_quick_img).pack(side="left", padx=5)
        ctk.CTkButton(i_btns, text="📋 Paste", width=100, command=self.paste_quick_img).pack(side="left", padx=5)
        
        self.qi_translate_btn = ctk.CTkButton(i_btns, text="Translate Image", font=("Inter", 14, "bold"), 
                                              fg_color=BTN_SUCCESS, hover_color=BTN_SUCCESS_HOVER,
                                              command=self.run_image_translate)
        self.qi_translate_btn.pack(side="left", padx=20)
        
        # Internal state for Image Mode
        self.current_qi_img = None 
        self.current_qi_path = None

        # Bind Paste globally (or specifically to this tab)
        self.bind("<Control-v>", lambda e: self.paste_quick_img())
        
        # SPRINT 39 Extension: Bind right-click Image Magic
        self.qi_preview_lbl.bind("<Button-3>", lambda e: self.show_image_context_menu(e, "quick"))
        

    def run_quick_translate(self):
        """Executes translation for the Quick Translate tab."""
        raw_text = self.qt_input.get("1.0", "end-1c").strip()
        if not raw_text or raw_text == "Paste or type text here and press Ctrl+Enter...":
            return
            
        src_lang = self.qt_src_lang.get()
        tgt_lang = self.qt_tgt_lang.get()
        
        self.qt_cancel_flag = False
        
        # SPRINT 83: Use robust unified translate_worker
        # Map nice names using the class-level dictionary
        src_info = self.languages.get(src_lang, {"trans": "auto"})
        tgt_info = self.languages.get(tgt_lang, {"trans": "en"})
        
        src_code = src_info.get("trans", "auto")
        tgt_code = tgt_info.get("trans", "en")
        
        self.qt_btn.configure(state="disabled", text="Translating...")
        self.qt_output.configure(state="normal")
        self.qt_output.delete("1.0", "end")
        self.qt_output.insert("1.0", "Translating...")
        self.qt_output.configure(state="disabled")
        
        def _translate_job():
            try:
                # SPRINT 84: Reuse the heavy-duty translate_worker for reliability
                res = translate_worker(
                    text=raw_text, 
                    translator_src=src_code, 
                    target_lang=tgt_code,
                    engine=self.selected_engine,
                    deepl_key=self.deepl_key,
                    openai_key=self.openai_key,
                    ollama_model=getattr(self, 'ollama_model', 'llama3'),
                    glossary=self.glossary,
                    cancel_check=lambda: getattr(self, "qt_cancel_flag", False)
                )
                
                if self.qt_cancel_flag: 
                    return
                    
                final_text = res.get("translated")
                if not final_text:
                    final_text = res.get("error", "Error: Translation failed.")
                
                self.schedule_ui_update(lambda: self._finalize_qt(final_text))
            except Exception as e:
                self.schedule_ui_update(lambda err=e: self._finalize_qt(f"Error: {err}"))
                
        threading.Thread(target=_translate_job, daemon=True).start()
        
    def _finalize_qt(self, result_text):
        """Standardizes the output for the Quick Translate tab."""
        print("[UI] Finalizing translation display...")
        self.qt_output.configure(state="normal")
        self.qt_output.delete("1.0", "end")
        
        # SPRINT 92: Broaden error detection to capture any engine failure for user advice
        res_str = str(result_text)
        error_keywords = ["429", "Too Many Requests", "405", "failed", "error", "exception", "sorry/index"]
        
        if any(kw.lower() in res_str.lower() for kw in error_keywords):
            final_msg = "429 Error: Please send the transcript in parts (e.g., one paragraph at a time).\n\nDetails: Google is rate-limiting or blocking the connection. Try switching the Engine to GPT-4o or DeepL in Settings ⚙️."
        else:
            final_msg = result_text

        # Tag for RTL if output is Arabic/Hebrew
        if self.qt_tgt_lang.get() in ["Arabic", "Hebrew"]:
            self.qt_output.insert("1.0", final_msg, "rtl")
            self.qt_output.tag_config("rtl", justify="right")
        else:
            self.qt_output.insert("1.0", final_msg)
            
        self.qt_output.configure(state="disabled")
        if hasattr(self, 'qt_btn'):
            self.qt_btn.configure(state="normal", text="Translate Text")

    def upload_quick_img(self):
        """Allows user to pick an image for Quick Translate."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if file_path:
            try:
                img = Image.open(file_path)
                self.current_qi_path = file_path
                self._update_qi_preview(img)
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not open image: {e}")

    def paste_quick_img(self, file_path=None):
        """Grabs image from clipboard (or path) and shows preview."""
        try:
            if file_path and os.path.exists(file_path):
                img = Image.open(file_path)
                self.current_qi_path = file_path
                self._update_qi_preview(img)
                return

            from PIL import ImageGrab
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                self.current_qi_img = img
                self.current_qi_path = None # It's in memory
                self._update_qi_preview(img)
            elif isinstance(img, list) and len(img) > 0:
                # Handle file paths in clipboard
                f_path = img[0]
                if os.path.exists(f_path):
                    self.current_qi_path = f_path
                    self._update_qi_preview(Image.open(f_path))
            else:
                # SPRINT 67: Check if text is present instead of just erroring
                # If there's text, we want to allow the normal paste to happen if a textbox is focused
                try: 
                    text_content = self.clipboard_get()
                    if text_content: return # Silently exit, letting standard paste handle it
                except: pass
                
                messagebox.showinfo("Clipboard", "No image found in clipboard.")
        except Exception as e:
            print(f"Clipboard/Drop Error: {e}")

    def _update_qi_preview(self, img):
        """Standardizes and displays the image preview in the UI."""
        self.current_qi_img = img
        # Resize for preview (Max 400x400)
        preview_img = img.copy()
        preview_img.thumbnail((350, 400))
        
        ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)
        self.qi_preview_lbl.configure(image=ctk_img, text="")
        self.qi_preview_lbl.image = ctk_img # Keep reference

    def run_image_translate(self):
        """Executes OCR + Translation for the current Quick Image."""
        if not self.current_qi_img:
            messagebox.showwarning("No Image", "Please upload or paste an image first.")
            return
            
        src_lang = self.qi_src_lang.get()
        tgt_lang = self.qi_tgt_lang.get()
        
        # UI Feedback
        self.qi_translate_btn.configure(state="disabled", text="OCRing...")
        for box in [self.qi_transcript, self.qi_translation]:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", "Analyzing image...")
            box.configure(state="disabled")

        def _image_job():
            try:
                # 1. OCR Configuration (SPRINT 83: Unified mapping)
                lang_info = self.languages.get(src_lang, {"ocr": "eng", "trans": "auto"})
                ocr_lang = lang_info.get("ocr", "eng")
                src_code = lang_info.get("trans", "auto")
                
                # SPRINT 83: Robust Environment Handling (consistent with ocr_worker)
                import os, cv2, numpy as np
                t_cmd = getattr(self, "tesseract_cmd", "tesseract")
                pytesseract.pytesseract.tesseract_cmd = t_cmd
                tess_dir = os.path.dirname(t_cmd)
                os.environ["TESSDATA_PREFIX"] = os.path.join(tess_dir, "tessdata")

                # 2. Advanced Preprocessing
                # Convert PIL to CV2
                cv_img = cv2.cvtColor(np.array(self.current_qi_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                denoised = cv2.medianBlur(gray, 3)
                thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                
                # Mask out speckles (consistent with ocr_worker)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh), connectivity=8)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 10:
                        thresh[labels == i] = 255
                
                process_img = Image.fromarray(thresh)
                
                # 3. Dynamic PSM Configuration
                psm_cfg = '--oem 1 --psm 3'
                if any(x in ocr_lang for x in ["ara", "heb", "yid"]):
                    psm_cfg = '--oem 1 --psm 6' # Block mode for joined scripts
                
                extracted_text = pytesseract.image_to_string(process_img, lang=ocr_lang, config=psm_cfg).strip()
                extracted_text = self.clean_ocr_text(extracted_text)
                
                if not extracted_text:
                    self.schedule_ui_update(lambda: messagebox.showinfo("OCR Result", "No text detected in image."))
                    self.schedule_ui_update(lambda: self.qi_translate_btn.configure(state="normal", text="Translate Image"))
                    return

                # UI Update (Transcript)
                self.schedule_ui_update(lambda text=extracted_text: [
                    self.qi_transcript.configure(state="normal"),
                    self.qi_transcript.delete("1.0", "end"),
                    self.qi_transcript.insert("1.0", text),
                    self.qi_transcript.configure(state="normal"), # SPRINT 88: Allow editing
                    self.qi_translation.configure(state="normal"),
                    self.qi_translation.delete("1.0", "end"),
                    self.qi_translation.insert("1.0", "[Translating...]"),
                    self.qi_translation.configure(state="disabled"),
                    self.qi_translate_btn.configure(text="Translating...")
                ])

                # SPRINT 84: Map nice names using labels
                # Map nice names using the class-level dictionary
                src_info = self.languages.get(src_lang, {"ocr": "eng", "trans": "auto"})
                tgt_info = self.languages.get(tgt_lang, {"trans": "en"})
                src_code = src_info.get("trans", "auto")
                tgt_code = tgt_info.get("trans", "en")

                # 4. Standardized Translation
                res = translate_worker(
                    text=extracted_text, 
                    translator_src=src_code, 
                    target_lang=tgt_code,
                    engine=self.selected_engine,
                    deepl_key=self.deepl_key,
                    openai_key=self.openai_key,
                    ollama_model=getattr(self, 'ollama_model', 'llama3'),
                    glossary=self.glossary
                )
                
                def _done(final_text):
                    self.qi_translation.configure(state="normal")
                    self.qi_translation.delete("1.0", "end")
                    self.qi_translation.insert("1.0", final_text)
                    self.qi_translation.configure(state="disabled")
                    self.qi_translate_btn.configure(state="normal", text="Translate Image")

                # SPRINT 86: Robust Error Check
                final_val = res.get("translated")
                if not final_val:
                    final_val = res.get("error", "Error during translation.")
                
                def _wrap():
                    _done(final_val)
                    # SPRINT 88: Final safety - ensure enabled for editing
                    self.qi_transcript.configure(state="normal")
                    
                self.schedule_ui_update(_wrap)

            except Exception as e:
                self.schedule_ui_update(lambda err=e: messagebox.showerror("Error", f"Image translation failed: {err}"))
                self.schedule_ui_update(lambda: self.qi_translate_btn.configure(state="normal", text="Translate Image"))

        threading.Thread(target=_image_job, daemon=True).start()

    def open_settings(self):
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Text Settings")
        settings_win.geometry("300x250")
        settings_win.attributes("-topmost", True)
        
        # Font Family
        ctk.CTkLabel(settings_win, text="Font Family:").pack(pady=(20, 5))
        fonts = ["Times New Roman", "Helvetica", "Arial", "Courier New", "Georgia", "Verdana"]
        font_menu = ctk.CTkOptionMenu(settings_win, values=fonts, command=self.update_font_family)
        font_menu.set(self.font_family)
        font_menu.pack(pady=5)
        
        # Font Size
        ctk.CTkLabel(settings_win, text="Font Size:").pack(pady=(10, 5))
        size_slider = ctk.CTkSlider(settings_win, from_=10, to=30, number_of_steps=20, command=self.update_font_size)
        size_slider.set(self.font_size)
        size_slider.pack(pady=5)

        # Bookmarks List
        if self.bookmarks:
            ctk.CTkLabel(settings_win, text="Jump to Bookmark:").pack(pady=(10, 5))
            bm_vals = [f"Page {i+1}" for i in self.bookmarks]
            bm_menu = ctk.CTkOptionMenu(settings_win, values=bm_vals, 
                                        command=lambda v: [self.goto_bookmark(v), settings_win.destroy()])
            bm_menu.set("Select Bookmark")
            bm_menu.pack(pady=5)

    def goto_bookmark(self, val):
        page_num = int(val.split(" ")[1]) - 1
        self.current_page_idx = page_num
        self.render_page(page_num)

    def update_font_family(self, choice):
        self.font_family = choice
        self.apply_font_settings()
        
    def update_font_size(self, value):
        self.font_size = int(value)
        self.apply_font_settings()
        
    def on_file_drop(self, files):
        """Entry point for windnd file drops. Handles both PDFs and Images based on context."""
        # 🟢 DEEP DIAGNOSTIC: Show immediate feedback to confirm the event arrived!
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        admin_warning = "\n\n(Note: App is running as Admin - Ensure Explorer is too!)" if is_admin else ""
        
        # SPRINT 101: Log the packet arrival
        count = len(files) if files else 0
        print(f"[UI] Drop Event: Received {count} file(s) from Windows. Admin: {is_admin}")
        
        if not files: return
        
        import sys
        file_path = ""
        # 1. Decode Path (Robust)
        try:
            raw = files[0]
            print(f"[UI] Raw drop data: {raw}")
            
            # If force_unicode=True is working, raw is already a string.
            if isinstance(raw, str):
                file_path = raw
            elif isinstance(raw, bytes):
                encodings = ['utf-8', 'mbcs', sys.getfilesystemencoding(), 'cp1252']
                for enc in encodings:
                    try:
                        file_path = raw.decode(enc)
                        if os.path.exists(file_path):
                            break
                    except:
                        continue
            else:
                file_path = str(raw)
            
            # Final check for mangled paths (????)
            if "?" in file_path and not os.path.exists(file_path):
                # If we still have question marks and file isn't found, it's an encoding failure
                print(f"[UI] ERROR: Path contains unresolvable Unicode characters: {file_path}")
                messagebox.showerror("Unicode Error", f"Cannot load file with special characters.\nPath received: {file_path}\n\nTry renaming the file to English or disabling Administrator mode.")
                return

        except Exception as e:
            messagebox.showerror("Drop Error", f"Could not read the dropped file path.\n\nDetail: {e}")
            return

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Not Found", f"The dropped file path is invalid or inaccessible:\n{file_path}")
            return

        ext = os.path.splitext(file_path.lower())[1]
        
        # 2. Image Drop (If in Quick Translate Tab)
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
            if self.tab_view.get() == "Quick Translate":
                print(f"[UI] Image dropped: {file_path}")
                self.paste_quick_img(file_path) # Adapted to take path
            else:
                msg = f"Image dropped: {os.path.basename(file_path)}\n\n"
                msg += "Please switch to the 'Quick Translate' tab first to process images."
                messagebox.showinfo("Image Dropped", msg)
        
        # 3. PDF Drop (Main document)
        elif ext == '.pdf':
            print(f"[UI] PDF dropped: {file_path}")
            # Ensure we are on the PDF tab
            self.tab_view.set("PDF Translator")
            self.load_new_pdf(file_path)
        else:
            messagebox.showwarning("Unsupported File", f"Bilingual Translator Pro does not support {ext} files.\n\nPlease drop a PDF or a common Image format.")
