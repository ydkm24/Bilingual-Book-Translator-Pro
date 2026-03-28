"""
cloud.py - Cloud, Authentication, Library, and Social Hub methods.
Extracted from main.py during The Great Refactor as a mixin class.
"""
import os
import json
import time
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, cast
from PIL import Image, ImageTk

from utils import (
    APP_VERSION, humanize_error, resource_path, get_app_path,
    PAPER_BG, TEXT_COLOR, TEXT_DIM, HEADER_BG, ACCENT_COLOR, ACCENT_HOVER,
    PAPER_SHEET_BG, BORDER_COLOR, CARD_BG, FRAME_BG,
    BTN_PRIMARY, BTN_PRIMARY_HOVER, BTN_SECONDARY, BTN_SECONDARY_HOVER,
    INPUT_BG, BTN_DANGER, BTN_DANGER_HOVER, BTN_SUCCESS, BTN_SUCCESS_HOVER,
    BTN_WARNING, BTN_WARNING_HOVER,
)


class CloudMixin:
    """Mixin class providing all cloud, authentication, library, and social hub methods."""

    def handle_avatar_click(self):
        if not self.current_user:
            self.show_auth_window()
        else:
            self.show_account_dash()
            
    def update_avatar_ui(self):
        """Updates the top right avatar button with the user's PFP or default icon."""
        if self.current_user:
            username = self.current_username or "User"
            if getattr(self, "avatar_url", None):
                # Load small 32x32 image
                img = self.get_cached_avatar(self.current_user["id"], self.avatar_url, size=(32, 32))
                if img:
                    self.avatar_btn.configure(image=img, text=username, compound="top")
                else:
                    self.avatar_btn.configure(image=None, text=f"👤\n{username}")
            else:
                self.avatar_btn.configure(image=None, text=f"👤\n{username}")
        else:
            self.avatar_btn.configure(image=None, text="👤 Login")
            
    def show_account_dash(self):
        if not self.supabase or not self.current_user: return
        
        dash = ctk.CTkToplevel(self)
        dash.title("Account Dashboard")
        dash.geometry("400x500")
        dash.transient(self)
        dash.grab_set()
        dash.configure(fg_color=PAPER_BG)

        # Header
        name = self.current_username or "User"
        if getattr(self, "verified_badge", False):
            name += " ✓"
            
        header = ctk.CTkLabel(dash, text=f"Welcome, {name}!", font=("Inter", 20, "bold"), text_color=TEXT_COLOR)
        header.pack(pady=(30, 10))
        
        # SPRINT 42: Bio Section
        bio_text = getattr(self, "current_bio", "") or "No bio set."
        self.dash_bio_lbl = ctk.CTkLabel(dash, text=bio_text, font=("Inter", 12, "italic"), text_color=TEXT_DIM, wraplength=350)
        self.dash_bio_lbl.pack(pady=(0, 20))

        # Avatar Display
        self.dash_avatar_lbl = ctk.CTkLabel(dash, text="👤", font=("Inter", 60), text_color=ACCENT_COLOR, fg_color=CARD_BG, corner_radius=10, width=120, height=120)
        self.dash_avatar_lbl.pack(pady=10)
        
        if getattr(self, "avatar_url", None):
            img = self.get_cached_avatar(self.current_user["id"], self.avatar_url, size=(100, 100))
            if img:
                self.dash_avatar_lbl.configure(image=img, text="")
            else:
                self.dash_avatar_lbl.configure(text="[Avatar Set]", font=("Inter", 14))

        # Edit Bio btn
        bio_btn = ctk.CTkButton(dash, text="✏ Edit Bio", command=self.edit_bio_dialog, fg_color="transparent", hover_color=BTN_SECONDARY_HOVER, text_color=ACCENT_COLOR, border_width=1, border_color=ACCENT_COLOR)
        bio_btn.pack(pady=(0, 20))

        # Update Avatar btn
        update_btn = ctk.CTkButton(dash, text="Update Profile Picture", command=self.upload_avatar, fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR)
        update_btn.pack(pady=(0, 20))

        # Inbox Button
        inbox_btn = ctk.CTkButton(dash, text="📬 Access Inbox (Requests)", command=self.show_inbox, height=40, font=("Inter", 14, "bold"))
        inbox_btn.pack(fill="x", padx=40, pady=10)

        # Logout Button
        logout_btn = ctk.CTkButton(dash, text="Logout", command=lambda: self.logout(dash), fg_color=BTN_DANGER, hover_color=BTN_DANGER_HOVER, text_color="#FFFFFF")
        logout_btn.pack(side="bottom", pady=20)

    def edit_bio_dialog(self):
        if not self.supabase or not self.current_user: return
        dialog = ctk.CTkInputDialog(text="Enter your new bio (max 150 chars):", title="Edit Bio")
        new_bio = dialog.get_input()
        if new_bio is not None:
            try:
                # SPRINT 42: We use a broader Exception block because older DBs might not have the col yet
                self.supabase.table("profiles").update({"bio": new_bio[:150]}).eq("id", self.current_user["id"]).execute()
                self.current_bio = new_bio[:150]
                if hasattr(self, "dash_bio_lbl") and self.dash_bio_lbl.winfo_exists():
                    self.dash_bio_lbl.configure(text=self.current_bio)
                messagebox.showinfo("Success", "Bio updated successfully!")
            except Exception as e:
                err_msg = str(e)
                if "column \"bio\" of relation \"profiles\" does not exist" in err_msg:
                    messagebox.showerror("Database Needs Update", "Run SPRINT 42 MIgration in setup_supabase.sql to enable Bios.")
                else:
                    messagebox.showerror("Update Error", f"Failed to update bio: {e}")

    def upload_avatar(self):
        file_path = filedialog.askopenfilename(title="Select Profile Picture", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path: return
        
        try:
            import time
            from PIL import Image
            
            # Compress image before upload
            img = Image.open(file_path)
            img.thumbnail((256, 256)) # resize to max 256x256
            
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            file_name = f"{self.current_user['id']}_{int(time.time())}.png"
            
            # Upload to Supabase Storage
            self.supabase.storage.from_("avatars").upload(file_name, img_bytes, {"content-type": "image/png"})
            
            # Get public URL
            public_url = self.supabase.storage.from_("avatars").get_public_url(file_name)
            
            # Update DB
            self.supabase.table("profiles").update({"avatar_url": public_url}).eq("id", self.current_user["id"]).execute()
            
            self.avatar_url = public_url
            
            # Immediately refresh UI
            img = self.get_cached_avatar(self.current_user["id"], public_url, size=(100, 100), force_refresh=True)
            if img: self.dash_avatar_lbl.configure(image=img, text="")
            self.update_avatar_ui()
            
            messagebox.showinfo("Success", "Profile picture updated!")
            
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to upload avatar: {e}")

    def logout(self, window):
        self.current_user = None
        self.current_username = None
        self.avatar_url = None
        self.is_admin = False
        if os.path.exists("session.json"):
            os.remove("session.json")
        self.update_avatar_ui()
        self.refresh_library()
        window.destroy()

    # --- SPRINT 34: Auth Methods ---
    def load_auth_session(self) -> Optional[Dict[str, str]]:
        path = get_app_path("session.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except: pass
        return None
        
    def save_auth_session(self, session) -> None:
        if not session: return
        try:
            path = get_app_path("session.json")
            with open(path, "w") as f:
                json.dump({"access_token": session.access_token, "refresh_token": session.refresh_token}, f)
        except Exception as e:
            print(f"Failed to save session: {e}")
            
    def fetch_username(self):
        if not self.supabase or not self.current_user: return
        try:
            # Join with profiles to get username
            res = self.supabase.table("profiles").select("*").eq("id", self.current_user["id"]).execute()
            if res.data:
                prof = res.data[0]
                self.current_username = prof.get('username')
                self.is_admin = prof.get('is_admin', False)
                self.avatar_url = prof.get('avatar_url')
                self.verified_badge = prof.get('verified_badge', False)
                self.current_bio = prof.get('bio', '')
        except Exception as e:
            print(f"Failed to fetch username: {e}")

    def show_auth_window(self):
        """Displays a modal Login/Register window."""
        auth_win = ctk.CTkToplevel(self)
        auth_win.title("Welcome to Translator Pro")
        auth_win.geometry("400x550")
        auth_win.grab_set()
        auth_win.attributes('-topmost', True)
        
        ctk.CTkLabel(auth_win, text="Translator Pro Cloud", font=("Inter", 24, "bold"), text_color=ACCENT_COLOR).pack(pady=(40, 10))
        ctk.CTkLabel(auth_win, text="Login to access the Global Library", font=("Inter", 12)).pack(pady=(0, 10))
        
        self.auth_error_lbl = ctk.CTkLabel(auth_win, text="", text_color="#E74C3C", font=("Inter", 12, "bold"))
        self.auth_error_lbl.pack(pady=(0, 10))
        
        tab_view = ctk.CTkTabview(auth_win, width=320)
        tab_view.pack(padx=20, pady=5, fill="both", expand=True)
        
        tab_login = tab_view.add("Login")
        tab_reg = tab_view.add("Register")
        
        # --- LOGIN TAB ---
        ctk.CTkLabel(tab_login, text="Email or Username:").pack(anchor="w", padx=20, pady=(10, 0))
        log_email = ctk.CTkEntry(tab_login, width=280)
        log_email.pack(padx=20, pady=5)
        
        ctk.CTkLabel(tab_login, text="Password:").pack(anchor="w", padx=20, pady=(10, 0))
        log_pass = ctk.CTkEntry(tab_login, width=280, show="*")
        log_pass.pack(padx=20, pady=5)
        
        def _do_login():
            login_id = log_email.get().strip()
            pw = log_pass.get().strip()
            if not login_id or not pw: return
            
            log_btn.configure(state="disabled", text="Logging in...")
            self.auth_error_lbl.configure(text="")
            try:
                em = login_id
                if "@" not in login_id:
                    # Attempt to resolve username to email via profiles table
                    user_res = self.supabase.table("profiles").select("email").eq("username", login_id).execute()
                    if user_res.data and user_res.data[0].get("email"):
                        em = user_res.data[0]["email"]
                    else:
                        raise Exception("Username not found or has no linked email. Please login with Email.")
                        
                res = self.supabase.auth.sign_in_with_password({"email": em, "password": pw})
                if res and res.user and res.session:
                    self.current_user = {"id": res.user.id, "email": res.user.email}
                    
                    # Backfill email if missing in profiles for future username logins
                    try:
                        self.supabase.table("profiles").update({"email": res.user.email}).eq("id", res.user.id).execute()
                    except: pass
                    
                    self.save_auth_session(res.session)
                    self.fetch_username()
                    self.update_avatar_ui()
                    auth_win.destroy()
                    self.refresh_library() # Refresh to show permissions
            except Exception as e:
                err_msg = str(e)
                if "Invalid login" in err_msg: err_msg = "Invalid email or password."
                self.auth_error_lbl.configure(text=err_msg)
            finally:
                log_btn.configure(state="normal", text="Login")
                
        log_btn = ctk.CTkButton(tab_login, text="Login", command=_do_login, width=280)
        log_btn.pack(pady=30)
        
        # --- REGISTER TAB ---
        ctk.CTkLabel(tab_reg, text="Username:").pack(anchor="w", padx=20, pady=(10, 0))
        reg_user = ctk.CTkEntry(tab_reg, width=280)
        reg_user.pack(padx=20, pady=5)
        
        ctk.CTkLabel(tab_reg, text="Email:").pack(anchor="w", padx=20, pady=(10, 0))
        reg_email = ctk.CTkEntry(tab_reg, width=280)
        reg_email.pack(padx=20, pady=5)
        
        ctk.CTkLabel(tab_reg, text="Password:").pack(anchor="w", padx=20, pady=(10, 0))
        reg_pass = ctk.CTkEntry(tab_reg, width=280, show="*")
        reg_pass.pack(padx=20, pady=5)
        
        def _do_register():
            usr = reg_user.get().strip()
            em = reg_email.get().strip()
            pw = reg_pass.get().strip()
            
            if not usr or not em or len(pw) < 6:
                self.auth_error_lbl.configure(text="Please fill all fields. Password must be 6+ chars.")
                return
                
            reg_btn.configure(state="disabled", text="Registering...")
            self.auth_error_lbl.configure(text="")
            try:
                # 1. Sign up
                res = self.supabase.auth.sign_up({"email": em, "password": pw})
                if res and res.user:
                    user_id = res.user.id
                    # 2. Create Profile
                    try:
                        self.supabase.table("profiles").insert({"id": user_id, "username": usr, "email": em}).execute()
                    except Exception as prof_err:
                        print(f"Profile creation err: {prof_err}")
                        
                    # 3. Direct Login
                    login_res = self.supabase.auth.sign_in_with_password({"email": em, "password": pw})
                    if login_res and login_res.user and login_res.session:
                        self.current_user = {"id": user_id, "email": em}
                        self.current_username = usr
                        self.save_auth_session(login_res.session)
                        self.update_avatar_ui()
                        auth_win.destroy()
                        self.refresh_library()
            except Exception as e:
                self.auth_error_lbl.configure(text=str(e))
            finally:
                reg_btn.configure(state="normal", text="Create Account")
                
        reg_btn = ctk.CTkButton(tab_reg, text="Create Account", command=_do_register, width=280, fg_color="#4CAF50", hover_color="#45A049")
        reg_btn.pack(pady=20)
        
        # --- Skip Option ---
        ctk.CTkButton(auth_win, text="Skip for now (Local Mode Only)", fg_color="transparent", 
                      hover_color="#333333", command=auth_win.destroy).pack(pady=10)

    def build_library_ui(self):
        """Builds the nested UI for the Global Library tab."""
        # Main container
        container = ctk.CTkFrame(self.tab_library, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header with Search and Refresh
        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))

        self.lib_search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(header, placeholder_text="Search Library...", 
                                    textvariable=self.lib_search_var, width=250)
        search_entry.pack(side="left", padx=(0, 10))
        search_entry.bind("<KeyRelease>", lambda e: self.refresh_library())
        
        self.lib_filter_var = ctk.StringVar(value="Public Library")
        filter_seg = ctk.CTkSegmentedButton(header, values=["My Books", "Public Library"], variable=self.lib_filter_var, command=lambda e: self.refresh_library())
        filter_seg.pack(side="left", padx=10)

        refresh_btn = ctk.CTkButton(header, text="↻ Refresh", width=100, command=self.refresh_library)
        refresh_btn.pack(side="right")

        # Scrollable area for books
        self.lib_scroll = ctk.CTkScrollableFrame(container, fg_color=PAPER_BG)
        self.lib_scroll.pack(fill="both", expand=True)

        # Initial Load
        self.refresh_library()

    def refresh_library(self):
        """SPRINT 33/40: Fetches books from Supabase with account-based sync and permission checks."""
        if not self.supabase: return

        # Clear existing
        for widget in self.lib_scroll.winfo_children():
            widget.destroy()

        search_term = self.lib_search_var.get().lower()

        try:
            # Query Logic
            show_mode = self.lib_filter_var.get() if hasattr(self, "lib_filter_var") else "Public Library"
            
            try:
                if show_mode == "My Books" and getattr(self, "current_user", None):
                    # ACCOUNT SYNC: Pull everything owned by THIS account
                    res = self.supabase.table("books").select("*").eq("owner_id", self.current_user["id"]).order("created_at", desc=True).execute()
                else:
                    # PUBLIC LIBRARY: Pull everything marked public
                    res = self.supabase.table("books").select("*").eq("is_public", True).order("created_at", desc=True).execute()
            except Exception as e:
                err_msg = str(e)
                if "column books.owner_id does not exist" in err_msg or "column books.active_editor_id does not exist" in err_msg or "42703" in err_msg:
                    # SPRINT 37 / SPRINT 43 Fallback: If migration hasn't run, pull standard columns to prevent crashing 
                    res = self.supabase.table("books").select("id, title, total_pages, language, is_public, created_at").eq("is_public", True).order("created_at", desc=True).execute()
                    self.schedule_ui_update(lambda: messagebox.showwarning("Database Needs Update", "Please run the latest SQL migrations in setup_supabase.sql to enable cloud sync ownership and collab locks."))
                else:
                    self.status_label.configure(text=f"Connection error: {err_msg[:50]}...")
                    return
            
            books = res.data
            if not books:
                ctk.CTkLabel(self.lib_scroll, text="No books found in this view.").pack(pady=40)
                return
                
            local_cache_dir = get_app_path(".translator_cache")
            local_caches = set(os.listdir(local_cache_dir)) if os.path.exists(local_cache_dir) else set()

            # Group by Category
            raw_folders: Dict[str, List[Any]] = {}
            for book in books:
                if search_term and search_term not in book['title'].lower() and search_term not in book.get('language', '').lower():
                    continue
                
                cat_key = str(book.get('category', 'Uncategorized') or 'Uncategorized')
                if cat_key not in raw_folders: raw_folders[cat_key] = []
                raw_folders[cat_key].append(book)

            folders = cast(Dict[str, List[Any]], raw_folders)
            if not folders:
                ctk.CTkLabel(self.lib_scroll, text="No matches found.").pack(pady=40)
                return

            # Render Folders (Explorer Style Grid)
            sorted_cats = sorted(folders.keys())
            
            def render_chunk(cat_idx, book_idx):
                if cat_idx >= len(sorted_cats) or not self.winfo_exists():
                    return
                
                category = sorted_cats[cat_idx]
                cat_books = folders[category]
                
                if book_idx == 0:
                    cat_frame = ctk.CTkFrame(self.lib_scroll, fg_color=FRAME_BG, corner_radius=10)
                    cat_frame.pack(pady=10, fill="x", padx=10)

                    cat_header = ctk.CTkLabel(cat_frame, text=f"📂 {category}", 
                                              font=("Inter", 16, "bold"), text_color=("#6200EE", "#BB86FC"), anchor="w")
                    cat_header.pack(fill="x", padx=15, pady=10)

                    book_grid = ctk.CTkFrame(cat_frame, fg_color="transparent")
                    book_grid.pack(fill="x", padx=10, pady=(0, 10))
                    self.current_book_grid = book_grid
                else:
                    book_grid = self.current_book_grid

                # Render batch
                batch_size = 15
                end_idx = min(book_idx + batch_size, len(cat_books))
                for i in range(book_idx, end_idx):
                    book = cat_books[i]
                    card = ctk.CTkFrame(book_grid, fg_color=CARD_BG, width=280, height=120)
                    card.grid(row=i // 3, column=i % 3, padx=10, pady=10, sticky="nsew")
                    book_grid.grid_columnconfigure(i % 3, weight=1)

                    is_owner = self.current_user and book.get('owner_id') == self.current_user['id']
                    is_locked = book.get('is_read_only', True)
                    
                    status_icon = "👤" if is_owner else "🔓" if not is_locked else "🔒"

                    info_frame = ctk.CTkFrame(card, fg_color="transparent")
                    info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

                    ctk.CTkLabel(info_frame, text=book['title'], font=("Inter", 13, "bold"), anchor="w", wraplength=180).pack(fill="x")
                    
                    uploader = book.get('owner_username') or "Anonymous"
                    # SPRINT 42: Join to check if owner is verified
                    # We need to fetch the profile. Since we don't have a direct join locally, we'll fetch individually or just try our best.
                    # We can fetch the badge if we have owner_id, but it's inefficient to do per-card. 
                    # For a proper fix, the SQL query in refresh_library should be updated. Let's do that in a later refactor.
                    # For now, let's just show the uploader name normally since we didn't join `profiles`.
                    sub_text = f"{book['language']} • {book.get('total_pages','?')} Pgs\nBy: {uploader}"
                    ctk.CTkLabel(info_frame, text=sub_text, font=("Inter", 11), text_color=TEXT_DIM, anchor="w", justify="left").pack(fill="x")

                    act_frame = ctk.CTkFrame(card, fg_color="transparent")
                    act_frame.pack(side="right", padx=10)

                    if not is_locked or is_owner:
                        # Dual Buttons for editable books
                        ctk.CTkButton(act_frame, text="📖 Read", width=70, height=28,
                                      fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR,
                                      command=lambda b=book: self.load_book_from_cloud(b, read_only=True)).pack(pady=2)
                        
                        ctk.CTkButton(act_frame, text="✏️ Edit", width=70, height=28,
                                      fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER,
                                      command=lambda b=book: self.load_book_from_cloud(b, read_only=False)).pack(pady=2)
                    else:
                        # Request flow for locked books
                        ctk.CTkButton(act_frame, text="Request Edit", width=70, height=28,
                                      fg_color=BTN_WARNING, hover_color=BTN_WARNING_HOVER, text_color="#121212",
                                      command=lambda b=book: self.request_edit_access(b['id'], b['title'], b['owner_id'])).pack(pady=2)
                        
                        ctk.CTkButton(act_frame, text="📖 Read Only", width=70, height=28,
                                      fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_COLOR,
                                      command=lambda b=book: self.load_book_from_cloud(b, read_only=True)).pack(pady=2)

                    if getattr(self, "is_admin", False) or is_owner:
                        ctk.CTkButton(act_frame, text="🗑", width=30, height=28, fg_color=BTN_DANGER, 
                                      command=lambda b=book['id']: self.delete_cloud_book(b, self)).pack(pady=2)

                    # SPRINT 40: Social Hub
                    social_frame = ctk.CTkFrame(act_frame, fg_color="transparent")
                    social_frame.pack(pady=2, fill="x")
                    
                    likes_count = book.get('likes', 0)
                    like_btn = ctk.CTkButton(social_frame, text=f"❤️ {likes_count}", width=60, height=28, fg_color=CARD_BG, hover_color=BTN_SECONDARY_HOVER, text_color="#E74C3C")
                    like_btn.configure(command=lambda b=book, btn=like_btn: self.like_book(b['id'], b.get('likes', 0), btn))
                    like_btn.pack(side="left", padx=(0, 2))
                    
                    ctk.CTkButton(social_frame, text="💬", width=30, height=28, fg_color=CARD_BG, hover_color=BTN_SECONDARY_HOVER, text_color=ACCENT_COLOR,
                                  command=lambda b=book: self.show_comments_dialog(b)).pack(side="right", padx=(2, 0), expand=True, fill="x")

                if end_idx < len(cat_books):
                    self.after(5, render_chunk, cat_idx, end_idx)
                else:
                    self.after(5, render_chunk, cat_idx + 1, 0)
                    
            render_chunk(0, 0)
        except Exception as e:
            msg = f"Error connecting to Cloud: {e}"
            self.schedule_ui_update(lambda m=msg: self.status_label.configure(text=m))
            ctk.CTkLabel(self.lib_scroll, text=msg, text_color="#E74C3C").pack(pady=40)

    # --- SPRINT 40: Social Hub Methods ---
    def like_book(self, book_id, current_likes, btn_widget):
        if not self.supabase or not getattr(self, "current_user", None):
            messagebox.showinfo("Login Required", "You must be logged in to like books.")
            return
            
        try:
            # SPRINT 79: Check if already liked (Toggle Toggle logic)
            res = self.supabase.table("book_likes").select("*").eq("book_id", book_id).eq("user_id", self.current_user['id']).execute()
            
            if res.data:
                # UNLIKE
                self.supabase.table("book_likes").delete().eq("book_id", book_id).eq("user_id", self.current_user['id']).execute()
                new_likes = max(0, current_likes - 1)
                self.supabase.table("books").update({"likes": new_likes}).eq("id", book_id).execute()
                btn_widget.configure(text=f"🤍  {new_likes}") # Added extra space
                btn_widget.configure(command=lambda b_id=book_id, l=new_likes, w=btn_widget: self.like_book(b_id, l, w))
            else:
                # LIKE
                self.supabase.table("book_likes").insert({"book_id": book_id, "user_id": self.current_user['id']}).execute()
                new_likes = current_likes + 1
                self.supabase.table("books").update({"likes": new_likes}).eq("id", book_id).execute()
                btn_widget.configure(text=f"❤️  {new_likes}") # Added extra space for stability
                btn_widget.configure(command=lambda b_id=book_id, l=new_likes, w=btn_widget: self.like_book(b_id, l, w))
        except Exception as e:
            err_msg = str(e)
            if "relation \"book_likes\" does not exist" in err_msg:
                 messagebox.showerror("Database Update Required", "The 'book_likes' table is missing. Please run the SPRINT 79 SQL in setup_supabase.sql.")
            else:
                 print(f"Failed to toggle like: {e}")

    def show_comments_dialog(self, book):
        if not self.supabase: return
        
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Comments: {book['title']}")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color=PAPER_BG)
        
        header = ctk.CTkLabel(dialog, text=f"Comments for {book['title'][:30]}...", font=("Inter", 18, "bold"), text_color=TEXT_COLOR)
        header.pack(pady=15)
        
        scroll = ctk.CTkScrollableFrame(dialog, fg_color=FRAME_BG)
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        def refresh_comments():
            for widget in scroll.winfo_children():
                widget.destroy()
                
            try:
                res = self.supabase.table("comments").select("*").eq("book_id", book['id']).order("created_at", desc=False).execute()
                comments = res.data
                
                if not comments:
                    ctk.CTkLabel(scroll, text="No comments yet. Be the first!", text_color=TEXT_DIM).pack(pady=20)
                else:
                    for c in comments:
                        c_frame = ctk.CTkFrame(scroll, fg_color=CARD_BG)
                        c_frame.pack(fill="x", pady=5, padx=5)
                        
                        top_bar = ctk.CTkFrame(c_frame, fg_color="transparent")
                        top_bar.pack(fill="x", padx=10, pady=(5, 0))
                        
                        ctk.CTkLabel(top_bar, text=c['username'], font=("Inter", 12, "bold"), text_color=ACCENT_COLOR).pack(side="left")
                        date_str = c.get('created_at', '')[:10]
                        ctk.CTkLabel(top_bar, text=date_str, font=("Inter", 10), text_color=TEXT_DIM).pack(side="right")
                        
                        ctk.CTkLabel(c_frame, text=c['content'], font=("Inter", 12), text_color=TEXT_COLOR, justify="left", wraplength=400, anchor="w").pack(fill="x", padx=10, pady=5)
                        
                        # Admin / Owner delete
                        if getattr(self, "current_user", None) and (self.current_user['id'] == c['user_id'] or getattr(self, "is_admin", False) or book['owner_id'] == self.current_user['id']):
                            ctk.CTkButton(c_frame, text="Delete", width=40, height=20, fg_color="transparent", hover_color=BTN_DANGER_HOVER, text_color=TEXT_DIM, command=lambda c_id=c['id']: delete_comment(c_id)).pack(anchor="e", padx=5, pady=(0, 5))
                            
            except Exception as e:
                err_msg = str(e)
                if "relation \"comments\" does not exist" in err_msg or "PGRST205" in err_msg:
                    ctk.CTkLabel(scroll, text="Error: 'comments' table not found.\nThis feature requires the Social Hub database setup.\n\nPlease run the SQL in 'setup_supabase.sql' to fix.", 
                                 text_color="#E74C3C", font=("Inter", 12, "bold")).pack(pady=20)
                else:
                    ctk.CTkLabel(scroll, text=f"Error loading comments: {e}", text_color="#E74C3C").pack(pady=20)
                    
        def delete_comment(c_id):
            if messagebox.askyesno("Delete Comment", "Are you sure you want to delete this comment?"):
                self.supabase.table("comments").delete().eq("id", c_id).execute()
                refresh_comments()

        def post_comment():
            if not getattr(self, "current_user", None):
                messagebox.showinfo("Login Required", "You must be logged in to comment.")
                return
            content = inp.get("1.0", "end-1c").strip()
            if not content: return
            
            try:
                self.supabase.table("comments").insert({
                    "book_id": book['id'],
                    "user_id": self.current_user['id'],
                    "username": self.current_username or "Anonymous",
                    "content": content
                }).execute()
                inp.delete("1.0", "end")
                refresh_comments()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to post comment: {e}")

        inp_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        inp_frame.pack(fill="x", padx=20, pady=10)
        
        inp = ctk.CTkTextbox(inp_frame, height=60, fg_color=INPUT_BG, font=("Inter", 12))
        inp.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        post_btn = ctk.CTkButton(inp_frame, text="Post", width=80, command=post_comment)
        post_btn.pack(side="right")
        
        refresh_comments()

    def open_library(self):
        """Legacy method for popup - now switches to tab."""
        self.tab_view.set("Global Library")
        self.refresh_library()

    def get_cached_avatar(self, user_id, avatar_url, size=(32, 32), force_refresh=False):
        import urllib.request
        from PIL import Image, ImageDraw
        cache_dir = get_app_path(".translator_cache/avatars")
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{user_id}_{size[0]}.png")
        
        if not os.path.exists(file_path) or force_refresh:
            try:
                # Need absolute reliable cache invalidation if force refreshing
                req = urllib.request.Request(avatar_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                    
                # Make it round
                img = Image.open(file_path).convert("RGBA")
                img = img.resize(size, Image.Resampling.LANCZOS)
                mask = Image.new("L", size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, size[0], size[1]), fill=255)
                img.putalpha(mask)
                img.save(file_path, "PNG")
            except Exception as e:
                print(f"Avatar fetch failed: {e}")
                return None
                
        try:
            return ctk.CTkImage(light_image=Image.open(file_path), dark_image=Image.open(file_path), size=size)
        except:
            return None

    def delete_cloud_book(self, book_id, window):
        """Allows user to delete their own uploaded books."""
        if not self.supabase: return
        if not messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this book from the cloud?"): return
        
        try:
            self.supabase.table("books").delete().eq("id", book_id).execute()
            messagebox.showinfo("Deleted", "Book removed from cloud.")
            self.refresh_library()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def publish_to_gallery(self):
        """SPRINT 33/40: Opens the Explorer-style Publish Manager."""
        if not self.supabase or not self.current_pdf_path: return
        
        # 1. Create Popup Window
        pub_win = ctk.CTkToplevel(self)
        pub_win.title("Publish Manager")
        pub_win.geometry("480x650") # Increased height for safety
        pub_win.grab_set() 
        
        main_container = ctk.CTkFrame(pub_win, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=25, pady=25)

        ctk.CTkLabel(main_container, text="Cloud Explorer", font=("Inter", 22, "bold"), text_color=TEXT_COLOR).pack(pady=(0, 20))
        
        # 2. Custom Title Input
        ctk.CTkLabel(main_container, text="Book Title:", font=("Inter", 12, "bold"), text_color=TEXT_COLOR).pack(anchor="w")
        title_var = ctk.StringVar(value=os.path.basename(self.current_pdf_path))
        ctk.CTkEntry(main_container, textvariable=title_var, width=400).pack(pady=(5, 15))
        
        # 3. Folder/Category Selection (Explorer Style)
        ctk.CTkLabel(main_container, text="Select Destination Folder:", font=("Inter", 12, "bold"), text_color=TEXT_COLOR).pack(anchor="w")
        
        folders_frame = ctk.CTkFrame(main_container, fg_color=FRAME_BG, height=180)
        folders_frame.pack(fill="x", pady=(5, 5))
        folders_frame.pack_propagate(False) # Force the height
        
        folders_list = ctk.CTkScrollableFrame(folders_frame, fg_color="transparent")
        folders_list.pack(fill="both", expand=True, padx=5, pady=5)

        # State for folder selection
        selected_folder = ctk.StringVar(value="Uncategorized")
        folder_btns = {}
        locally_added_folders = set()

        def _select_folder(name):
            selected_folder.set(name)
            for n, btn in folder_btns.items():
                if n == name:
                    btn.configure(fg_color=("#BB86FC", "#6200EE"), text_color="#121212")
                else:
                    btn.configure(fg_color="transparent", text_color=TEXT_COLOR)

        def _refresh_folders():
            for widget in folders_list.winfo_children(): widget.destroy()
            folder_btns.clear()
            
            try:
                res = self.supabase.table("books").select("category").execute()
                names = set(["Uncategorized"])
                if res.data:
                    for b in res.data:
                        if b.get("category"): names.add(b["category"])
                
                # Merge with local temporary folders
                for local_f in locally_added_folders:
                    names.add(local_f)
                
                for name in sorted(list(names)):
                    btn = ctk.CTkButton(folders_list, text=f"📂 {name}", anchor="w", 
                                        fg_color="transparent", text_color=TEXT_COLOR,
                                        hover_color=("#BB86FC", "#6200EE"), height=30,
                                        command=lambda n=name: _select_folder(n))
                    btn.pack(fill="x", pady=2)
                    folder_btns[name] = btn
                
                _select_folder(selected_folder.get())
            except: pass

        _refresh_folders()

        # Add New Folder Row
        new_folder_row = ctk.CTkFrame(main_container, fg_color="transparent")
        new_folder_row.pack(fill="x", pady=(0, 15))
        
        new_f_entry = ctk.CTkEntry(new_folder_row, placeholder_text="New folder name...", width=320)
        new_f_entry.pack(side="left")

        def _add_folder():
            name = new_f_entry.get().strip()
            if name:
                locally_added_folders.add(name)
                selected_folder.set(name)
                new_f_entry.delete(0, "end")
                _refresh_folders()

        add_btn = ctk.CTkButton(new_folder_row, text="+", width=50, font=("Inter", 16, "bold"),
                                fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
                                command=_add_folder)
        add_btn.pack(side="right", padx=(10, 0))

        # 4. Privacy & Collaboration Toggles
        ctk.CTkLabel(main_container, text="Cloud Settings:", font=("Inter", 12, "bold"), text_color=TEXT_COLOR).pack(anchor="w")
        
        settings_box = ctk.CTkFrame(main_container, fg_color=FRAME_BG)
        settings_box.pack(fill="x", pady=(5, 10))

        # Public Release Toggle
        is_public_var = ctk.BooleanVar(value=False)
        public_switch = ctk.CTkSwitch(settings_box, text="Release to Public Library", variable=is_public_var,
                                      progress_color=("#BB86FC", "#6200EE"), text_color=TEXT_COLOR)
        public_switch.pack(anchor="w", padx=15, pady=10)
        
        # Read-Only Toggle
        read_only_var = ctk.BooleanVar(value=True)
        read_only_switch = ctk.CTkSwitch(settings_box, text="Read-Only (Requires Edit Request)", variable=read_only_var,
                                         text_color=TEXT_COLOR)
        read_only_switch.pack(anchor="w", padx=15, pady=(0, 10))

        # 5. Submit Action
        sub_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        sub_frame.pack(fill="x", pady=(10, 0))

        def _execute_publish():
            # Force immediate save of whatever is currently on screen
            self.save_edits_to_cache()
            
            # SPRINT 42: Block publish if book is still processing to avoid uploading "Translation Pending"
            if None in self.all_page_data or self.active_executor is not None:
                messagebox.showwarning("Still Processing", "Please wait until the book has finished fully scanning and translating before publishing. Syncing now would upload empty or pending pages.")
                return
            
            pub_title = title_var.get().strip() or (os.path.basename(self.current_pdf_path) if self.current_pdf_path else "Untitled")
            pub_folder = selected_folder.get().strip() or "Uncategorized"
            is_read_only = read_only_var.get()
            is_public = is_public_var.get()
            
            pub_btn.configure(state="disabled", text="Syncing to Cloud...")
            
            def _job():
                book_id_or_err = self.sync_book_metadata(pub_title, pub_folder, is_read_only, is_public)
                if not book_id_or_err or (isinstance(book_id_or_err, str) and book_id_or_err.startswith("Error:")):
                    err_hint = book_id_or_err if isinstance(book_id_or_err, str) else "Unknown"
                    self.schedule_ui_update(lambda msg=humanize_error(err_hint): messagebox.showerror("Cloud Error", f"Sync failed.\n\nReason: {msg}"))
                    self.schedule_ui_update(lambda: pub_btn.configure(state="normal", text="Sync to Cloud"))
                    return
                book_id = book_id_or_err
                    
                try:
                    total_p = len(self.all_page_data)
                    img_success = 0
                    img_fail = 0
                    
                    for i, page_data in enumerate(self.all_page_data):
                        if not page_data or "page" not in page_data: continue
                        page_idx = page_data["page"]
                        
                        # Progress feedback
                        self.schedule_ui_update(lambda idx=i+1: pub_btn.configure(text=f"Uploading Page {idx}/{total_p}..."))
                        
                        self.supabase.table("pages").upsert({
                            "book_id": book_id,
                            "page_index": page_idx,
                            "original": page_data.get("original", ""),
                            "english": page_data.get("english", ""),
                            "literal": page_data.get("literal", ""),
                            "is_image": page_data.get("is_image", False),
                            "is_cover": page_data.get("is_cover", False),
                            "is_centered": page_data.get("is_centered", False),
                            "is_rtl_page": page_data.get("is_rtl_page", False)
                        }, on_conflict="book_id,page_index").execute()
                        
                        # Upload image
                        _cache_path = getattr(self, "cache_dir", "")
                        dyn_img_path = os.path.join(_cache_path, f"img_{page_idx}.jpg")
                        
                        if os.path.exists(dyn_img_path):
                            try:
                                with open(dyn_img_path, "rb") as f:
                                    img_bytes = f.read()
                                    self.supabase.storage.from_("book-images").upload(
                                        path=f"{str(book_id)}/img_{int(page_idx)}.jpg",
                                        file=img_bytes, 
                                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                                    )
                                img_success += 1
                            except Exception as e:
                                print(f"Image upload fail (Page {page_idx}): {e}")
                                img_fail += 1
                    
                    self.current_book_id = book_id
                    self.save_settings()
                    
                    if total_p > 0 and img_success == 0:
                        warn_msg = f"Sync partial: {total_p} pages synced, but 0 images uploaded.\n\n"
                        warn_msg += f"This happens if the local cache was cleared or the folder name changed.\n"
                        warn_msg += f"Current project folder: {_cache_path}\n\n"
                        warn_msg += "Try to 'Re-Scan & Translate' this book again to rebuild the images locally first."
                        self.schedule_ui_update(lambda m=warn_msg: messagebox.showwarning("Sync Warning", m))
                    else:
                        success_msg = f"Book synced successfully!\n- {total_p} pages synced\n- {img_success} images uploaded"
                        if img_fail > 0:
                            success_msg += f"\n- {img_fail} images failed (check console)"
                        self.schedule_ui_update(lambda m=success_msg: messagebox.showinfo("Success", m))
                    
                    self.schedule_ui_update(self.refresh_library) 
                    self.schedule_ui_update(pub_win.destroy)
                except Exception as e:
                    self.schedule_ui_update(lambda err=humanize_error(e): messagebox.showerror("Error", f"Failed: {err}"))
                    self.schedule_ui_update(lambda: pub_btn.configure(state="normal", text="Sync to Cloud"))
                    
            threading.Thread(target=_job, daemon=True).start()
            
        pub_btn = ctk.CTkButton(sub_frame, text="Sync to Cloud", height=45, font=("Inter", 16, "bold"),
                                fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER, text_color="#FFFFFF", 
                                command=_execute_publish)
        pub_btn.pack(fill="x")

    def force_cloud_sync(self):
        """Manually pushes local edits back to cloud."""
        if not self.supabase or not self.all_page_data or not self.current_pdf_path: return
        
        title = os.path.basename(self.current_pdf_path)
        self.sync_btn.configure(state="disabled", text="Syncing...")
        
        def _job():
            # Force immediate save of whatever is currently on screen
            self.save_edits_to_cache()
            
            # SPRINT 42: Block publish if book is still processing to avoid uploading "Translation Pending"
            if None in self.all_page_data or self.active_executor is not None:
                self.schedule_ui_update(lambda: messagebox.showwarning("Still Processing", "Please wait until the book has finished fully scanning and translating before syncing. Syncing now would upload empty or pending pages."))
                self.schedule_ui_update(lambda: self.sync_btn.configure(state="normal", text="⇪ Sync Edits"))
                return
            
            try:
                # sync_book_metadata now requires title
                book_id = self.sync_book_metadata(title)
                if not book_id:
                    self.schedule_ui_update(lambda: messagebox.showerror("Sync Failed", "Book not registered in cloud or sync failed."))
                    return
                
                # 1. Check permissions
                res = self.supabase.table("books").select("id, is_read_only, owner_id").eq("id", book_id).execute()
                if not res.data:
                    self.schedule_ui_update(lambda: messagebox.showerror("Sync Failed", "Book is not registered in the cloud."))
                    return
                    
                book_id = res.data[0]['id']
                is_read_only = res.data[0].get('is_read_only', True)
                
                if is_read_only:
                    # SPRINT 34: Check if they are the owner OR if they have an approved request
                    is_owner = self.current_user and res.data[0].get('owner_id') == self.current_user['id']
                    
                    if getattr(self, "is_admin", False):
                        is_owner = True # Override for admin
                        
                    if not is_owner:
                        # Check for approved request
                        allowed = False
                        if self.current_user:
                            try:
                                req_res = self.supabase.table("permission_requests").select("status").eq("book_id", book_id).eq("requester_id", self.current_user["id"]).eq("status", "approved").execute()
                                if req_res.data:
                                    allowed = True
                            except Exception as e:
                                print(f"Permission Check Error: {e}")
                                
                        if not allowed:
                            self.schedule_ui_update(lambda: messagebox.showwarning("Permission Denied", "This book is marked Read-Only. You must request edit access from the owner."))
                            return
                
                # 2. Push all cached edits to cloud
                for page_data in self.all_page_data:
                    if not page_data or "page" not in page_data: continue
                    page_idx = page_data["page"]
                    
                    self.supabase.table("pages").upsert({
                        "book_id": book_id,
                        "page_index": page_idx,
                        "original": page_data.get("original", ""),
                        "english": page_data.get("english", ""),
                        "literal": page_data.get("literal", "")
                    }, on_conflict="book_id,page_index").execute()
                    
                self.current_book_id = book_id
                self.save_settings()
                self.schedule_ui_update(lambda: messagebox.showinfo("Sync Complete", "Collaborative edits saved to cloud!"))
            except Exception as e:
                self.schedule_ui_update(lambda err=e: messagebox.showerror("Sync Error", str(err)))
            finally:
                self.schedule_ui_update(lambda: self.sync_btn.configure(state="normal", text="⇪ Sync Edits"))
                
        threading.Thread(target=_job, daemon=True).start()

    def load_book_from_cloud(self, book_info, read_only=False):
        """Downloads book metadata almost instantly. Images are pulled lazily as needed."""
        if not self.supabase: return
        
        if not self.current_user:
            messagebox.showinfo("Login Required", "Please log in to edit or pull books from the cloud.")
            self.show_auth_window()
            return

        self.status_label.configure(text=f"Syncing '{book_info['title']}'...")
        self.menu_bar.entryconfigure("Export", state="disabled")
        
        # Disable translation controls to prevent interference
        self.pause_btn.configure(state="disabled")
        self.rescan_btn.configure(state="disabled")
        
        def _download_worker():
            try:
                # 1. Setup local cache dir for metadata
                book_id = book_info['id']
                temp_cache = get_app_path(os.path.join(".translator_cache", book_info['title'].replace(".pdf", "")))
                os.makedirs(temp_cache, exist_ok=True)
                self.cache_dir = temp_cache
                
                self.current_book_id = book_id
                title = book_info['title']
                if not title.lower().endswith(".pdf"): title += ".pdf"
                self.current_pdf_path = os.path.join(get_app_path("Downloads"), title)
                
                # Fetch text metadata only (INSTANT)
                res = self.supabase.table("pages").select("page_index, original, english, literal, is_image, is_cover, is_centered, is_rtl_page").eq("book_id", book_id).order("page_index").execute()
                
                if not res.data:
                    self.schedule_ui_update(lambda: messagebox.showwarning("Empty Book", "This cloud book has no translated pages yet."))
                    self.schedule_ui_update(lambda: self.status_label.configure(text="Ready"))
                    return

                # SPRINT 11.1 (Lazy Update): We do NOT download images here.
                # We just populate the all_page_data skeleton.
                book_total = book_info.get('total_pages', max((p['page_index'] for p in res.data), default=0) + 1)
                new_page_data = [None] * book_total
                
                for p in res.data:
                    idx = p['page_index']
                    new_page_data[idx] = {
                        "page": idx,
                        "original": p['original'],
                        "english": p['english'],
                        "literal": p['literal'],
                        "is_image": p.get('is_image', False),
                        "is_cover": p.get('is_cover', False),
                        "is_centered": p.get('is_centered', False),
                        "is_rtl_page": p.get('is_rtl_page', False),
                        # Mark as cloud resource for lazy fetching
                        "cloud_id": book_id,
                        "cover_image_path": os.path.join(temp_cache, f"img_{idx}.jpg")
                    }
                
                # Update UI elements back on the main thread
                def _finalize_load():
                    self.all_page_data = new_page_data
                    self.total_pages = len(self.all_page_data)
                    self.current_page_idx = 0
                    
                    # Update tabs
                    self.tab_view.set("PDF Translator")
                    
                    # SPRINT 11.1: Force Read-Only if requested or if book is locked
                    self.read_only_mode = read_only or book_info.get('is_read_only', True)
                    session_read_only = self.read_only_mode
                    
                    # SPRINT 34: Permissions & Request Flow
                    current_uid = self.current_user.get('id') if self.current_user else None
                    is_owner = current_uid and book_info.get('owner_id') == current_uid
                    
                    if getattr(self, "is_admin", False):
                        is_owner = True # Override for admin
                        session_read_only = False
                        
                    # SPRINT 43: Real-Time Collaboration Lock Implementation
                    # 1. Check if an active editor exists and the lock is fresh (within 5 minutes)
                    active_editor_id = book_info.get('active_editor_id')
                    active_editor_time = book_info.get('active_editor_time')
                    lock_active = False
                    
                    if active_editor_id and active_editor_id != self.current_user.get('id'):
                        if active_editor_time:
                            import datetime
                            try:
                                # Supabase timestamps are ISO 8601 strings
                                lock_time = datetime.datetime.fromisoformat(active_editor_time.replace('Z', '+00:00'))
                                now = datetime.datetime.now(datetime.timezone.utc)
                                if (now - lock_time).total_seconds() < 300: # 5 minutes lock expiry
                                    lock_active = True
                            except Exception as e:
                                print(f"Time parsing error: {e}")
                                
                    if lock_active:
                        self.schedule_ui_update(lambda: messagebox.showwarning("Book Locked", "Another user is currently editing this book. You have been placed in Read-Only mode to prevent overwriting their work."))
                        self.read_only_mode = True
                        session_read_only = True
                        is_owner = False # Force read-only behavior even if owner
                        
                    elif not session_read_only or is_owner:
                        # 2. We are allowed to edit. Claim the lock if NOT explicitly in read mode
                        if not read_only:
                            try:
                                self.supabase.table("books").update({
                                    "active_editor_id": self.current_user["id"],
                                    "active_editor_time": "now()"
                                }).eq("id", book_info['id']).execute()
                                
                                # 3. Start Heartbeat Thread to keep lock alive
                                self.heartbeat_active = True
                                def _heartbeat():
                                    while getattr(self, "heartbeat_active", False) and getattr(self, "current_book_id", None) == book_info['id']:
                                        import time; time.sleep(60) # Ping every 1 minute
                                        if not getattr(self, "heartbeat_active", False): break
                                        try:
                                            self.supabase.table("books").update({"active_editor_time": "now()"}).eq("id", book_info['id']).execute()
                                        except: pass
                                threading.Thread(target=_heartbeat, daemon=True).start()
                            except Exception as e:
                                print(f"Failed to claim lock: {e}")

                    is_locked = book_info.get('is_read_only', True)
                    # If it's logically locked and we aren't the owner, show the request button
                    if is_locked and not is_owner and not lock_active:
                        self.sync_btn.configure(state="disabled") # Disable sync explicitly
                        self.publish_btn.configure(text="Request Edit Access", fg_color="#F39C12", hover_color="#D68910", text_color="#FFFFFF", state="normal")
                        
                        # Replace publish command with request command for this session
                        self.publish_btn.configure(command=lambda: self.request_edit_access(book_info['id'], book_info['title'], book_info.get('owner_id')))
                        self.publish_btn.pack(side="right", padx=10, pady=10)
                    elif lock_active:
                        # View-only state due to active editor (hide request button)
                        self.sync_btn.configure(state="disabled")
                        self.publish_btn.pack_forget()
                    elif is_owner and is_locked:
                        self.publish_btn.pack_forget() 
                        self.sync_btn.configure(state="normal") # Owner can sync their own locked book
                        # We don't unlock the UI boxes for the owner because `is_locked` is globally respected in `set_box_lock` below, unless we override it. Let's override for the owner.
                        is_locked = False 
                    else:
                        self.publish_btn.pack_forget() # Hide if unlocked or we own it (and it's not locked against us)
                        self.sync_btn.configure(state="normal")
                    
                    def set_box_lock(box, locked):
                        box.configure(state="normal")
                        if locked: box.configure(state="disabled")

                    set_box_lock(self.orig_text, session_read_only)
                    set_box_lock(self.lit_text, session_read_only)
                    set_box_lock(self.trans_text, session_read_only)

                    self.render_page(0)
                    
                    # Revealed specific buttons
                    self.sync_btn.pack(side="right", padx=5, pady=10)
                    self.sync_btn.configure(state="disabled" if session_read_only else "normal")
                    self.publish_btn.pack_forget()
                    self.menu_bar.entryconfigure("Export", state="normal")
                    self.pause_btn.configure(state="disabled" if session_read_only else "normal")

                    self.status_label.configure(text=f"Joined '{book_info['title']}' ({'READ ONLY' if session_read_only else 'EDITING'})")
                    # Force a refresh of the UI layout
                    self.update_idletasks()
                    
                self.schedule_ui_update(_finalize_load)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.schedule_ui_update(lambda err=humanize_error(e): messagebox.showerror("Cloud Error", f"Failed to pull book: {err}"))
                self.schedule_ui_update(lambda: self.status_label.configure(text="Ready"))
                
        # Start background thread
        threading.Thread(target=_download_worker, daemon=True).start()

    def request_edit_access(self, book_id, book_title, owner_id):
        """Sends a permission request to the book owner."""
        if not self.supabase or not self.current_user:
            messagebox.showwarning("Login Required", "You must be logged in to request access.")
            return
            
        if not owner_id:
            messagebox.showerror("Error", "This book has no assigned owner.")
            return
            
        # Check if already requested
        try:
            res = self.supabase.table("permission_requests").select("status").eq("book_id", book_id).eq("requester_id", self.current_user["id"]).execute()
            if res.data:
                status = res.data[0]['status']
                if status == 'pending':
                    messagebox.showinfo("Already Requested", "You have a pending request for this book.")
                    return
                elif status == 'approved':
                    messagebox.showinfo("Already Approved", "Your request is already approved. Please refresh the library.")
                    return
        except Exception as e:
            print(f"Check Request Error: {e}")
            
        # Insert new request
        try:
            self.supabase.table("permission_requests").insert({
                "book_id": book_id,
                "owner_id": owner_id,
                "requester_id": self.current_user["id"],
                "requester_username": self.current_username,
                "book_title": book_title
            }).execute()
            messagebox.showinfo("Request Sent", f"Edit access requested for '{book_title}'.\nThe owner will be notified.")
            self.publish_btn.configure(text="Request Pending", state="disabled")
        except Exception as e:
            messagebox.showerror("Request Failed", f"Failed to send request: {humanize_error(e)}")

    def show_inbox(self):
        """SPRINT 34: Shows incoming permission requests."""
        if not self.supabase or not self.current_user: return
        
        inbx = ctk.CTkToplevel(self)
        inbx.title("Publisher Inbox")
        inbx.geometry("500x400")
        inbx.transient(self)
        inbx.grab_set()
        inbx.configure(fg_color=PAPER_BG)
        
        ctk.CTkLabel(inbx, text="Incoming Edit Requests", font=("Inter", 18, "bold"), text_color=TEXT_COLOR).pack(pady=(20, 10))
        
        scroll = ctk.CTkScrollableFrame(inbx, width=450, height=300, fg_color="transparent")
        scroll.pack(pady=10, padx=20, fill="both", expand=True)
        
        try:
            # Get books owned by this user
            books_res = self.supabase.table("books").select("id, title").eq("owner_id", self.current_user["id"]).execute()
            if not books_res.data:
                ctk.CTkLabel(scroll, text="You have not published any books.", text_color=TEXT_DIM).pack(pady=20)
                return
                
            book_map = {b['id']: b['title'] for b in books_res.data}
            book_ids = list(book_map.keys())
            
            # Use 'in' filter to get requests for these books
            reqs = self.supabase.table("permission_requests").select("*").in_("book_id", book_ids).eq("status", "pending").execute()
            
            if not reqs.data:
                ctk.CTkLabel(scroll, text="No pending requests.", text_color=TEXT_DIM).pack(pady=20)
                return
                
            for req in reqs.data:
                # Build Request Card...
                req_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=8)
                req_card.pack(fill="x", pady=5)
                
                info = ctk.CTkFrame(req_card, fg_color="transparent")
                info.pack(side="left", padx=10, pady=10)
                
                b_title = book_map.get(req['book_id'], "Unknown Book")
                ctk.CTkLabel(info, text=f"From: {req['requester_username']}", font=("Inter", 12, "bold"), text_color=TEXT_COLOR).pack(anchor="w")
                ctk.CTkLabel(info, text=f"Book: {b_title}", font=("Inter", 11), text_color=TEXT_DIM).pack(anchor="w")
                
                acts = ctk.CTkFrame(req_card, fg_color="transparent")
                acts.pack(side="right", padx=10)
                
                def _approve(r_id=req['id']):
                    try:
                        self.supabase.table("permission_requests").update({"status": "approved"}).eq("id", r_id).execute()
                        inbx.destroy()
                        self.show_inbox() # Refresh
                        messagebox.showinfo("Success", "Request approved.")
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                        
                def _deny(r_id=req['id']):
                    try:
                        self.supabase.table("permission_requests").update({"status": "denied"}).eq("id", r_id).execute()
                        inbx.destroy()
                        self.show_inbox() # Refresh
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                        
                ctk.CTkButton(acts, text="Approve", width=60, fg_color=BTN_SECONDARY, hover_color="#4CAF50", command=_approve).pack(side="left", padx=5)
                ctk.CTkButton(acts, text="Deny", width=60, fg_color=BTN_DANGER, hover_color=BTN_DANGER_HOVER, command=_deny).pack(side="left")
                
        except Exception as e:
            err_msg = str(e)
            print(f"Inbox Fetch Error: {err_msg}")
            # Gracefully degrade to empty state for any database fetch errors to avoid scaring the user
            ctk.CTkLabel(scroll, text="No pending requests.", text_color=TEXT_DIM).pack(pady=40)


    def process_sync_queue(self):
        """Worker thread that handles cloud sync sequentially."""
        while True:
            item = None
            with self.sync_lock:
                if self.sync_queue:
                    item = self.sync_queue.pop(0)
                else:
                    self.sync_worker_active = False
                    break
            
            if item:
                try:
                    self.sync_page_to_cloud(item)
                except:
                    pass
        
        # Explicit cleanup after batch sync
        import gc
        gc.collect()

    def sync_book_metadata(self, book_title, category="Uncategorized", is_read_only=True, is_public=False):
        """Registers or updates book information in Supabase."""
        if not self.supabase or not self.current_user:
            return "Error: Not logged in"
        
        try:
            # Check for existing
            res = self.supabase.table("books").select("id").eq("title", book_title).eq("owner_id", self.current_user["id"]).execute()
            if res.data:
                book_id = res.data[0]["id"]
                self.supabase.table("books").update({
                    "category": category,
                    "is_read_only": is_read_only,
                    "is_public": is_public,
                    "total_pages": len(self.all_page_data)
                }).eq("id", book_id).execute()
                return book_id
                
            # Register new book
            new_book = {
                "title": book_title,
                "total_pages": len(self.all_page_data),
                "language": self.lang_menu.get(),
                "category": category,
                "is_read_only": is_read_only,
                "is_public": is_public,
                "owner_id": self.current_user["id"],
                "owner_username": self.current_username
            }
            res = self.supabase.table("books").insert(new_book).execute()
            return res.data[0]['id']
        except Exception as e:
            error_msg = str(e)
            print(f"Metadata Sync Error: {error_msg}")
            return f"Error: {error_msg}"

    def sync_page_to_cloud(self, page_data):
        """Uploads page JSON and any images to Supabase."""
        if not self.supabase: return
        
        book_id = self.sync_book_metadata()
        if not book_id: return
        
        page_num = page_data['page']
        try:
            # 1. Sync DB
            payload = {
                "book_id": book_id,
                "page_index": page_num,
                "original": page_data['original'],
                "english": page_data['english'],
                "literal": page_data['literal'],
                "is_image": page_data['is_image'],
                "is_cover": page_data['is_cover'],
                "is_centered": page_data.get('is_centered', False),
                "is_rtl_page": page_data.get('is_rtl_page', False)
            }
            
            # Upsert: Update if book_id+page_index exists
            self.supabase.table("pages").upsert(payload, on_conflict="book_id,page_index").execute()
            
            # 2. Sync Image to Storage
            dyn_img_path = os.path.join(self.cache_dir, f"img_{page_num}.jpg")
            if os.path.exists(dyn_img_path):
                file_name = f"{book_id}/img_{page_num}.jpg"
                try:
                    with open(dyn_img_path, 'rb') as f:
                        img_bytes = f.read()
                        self.supabase.storage.from_("book-images").upload(
                            file_name, img_bytes, file_options={"content-type": "image/png", "upsert": "true"}
                        )
                except Exception as img_e:
                    print(f"Failed to upload image for page {page_num}: {img_e}")
        except Exception as e:
            print(f"Cloud Sync Error (Page {page_num}): {e}")


    # ===========================
    # SPRINT 39: "Reader Magic"
    # ===========================
