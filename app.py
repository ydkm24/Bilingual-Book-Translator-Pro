import os
import sys
import eel
import json
from dotenv import load_dotenv
from supabase import create_client, Client, AuthApiError
from typing import Optional, Dict, Any
from tkinter import filedialog, Tk
from web_processing import WebTranslatorManager, LANGUAGES

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

print(f"Initializing Supabase for {SUPABASE_URL}...", flush=True)

if SUPABASE_URL and SUPABASE_KEY and "YOUR_SUPABASE" not in SUPABASE_URL:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase connected successfully.", flush=True)
    except Exception as e:
        print(f"Supabase Init Error: {e}", flush=True)

# State Management
current_user = None
translator = WebTranslatorManager()

def get_app_path(filename):
    """Helper to get path in the app's local data directory."""
    return os.path.join(BASE_DIR, filename)

# Initialize Eel with the web_ui folder
eel.init(os.path.join(BASE_DIR, "web_ui"))


# ============================================================
# EXPOSED PYTHON FUNCTIONS (callable from JS as eel.func())
# ============================================================

@eel.expose
def get_version():
    """Returns the current app version string."""
    return "v4.2.0"


@eel.expose
def get_app_info():
    """Returns basic app metadata for the Dashboard."""
    return {
        "status": "All Systems Nominal",
        "uptime": "99.9%",
        "build": "Stable-142",
        "last_update": "2026-03-20"
    }

# --- AUTHENTICATION API ---

@eel.expose
def check_session():
    """Checks if a valid session exists on startup."""
    global current_user
    if not supabase: return None
    
    session_path = get_app_path("session.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
                res = supabase.auth.set_session(data['access_token'], data['refresh_token'])
                if res and res.user:
                    print(f"Session restored for: {res.user.email}", flush=True)
                    current_user = {"id": res.user.id, "email": res.user.email}
                    
                    # Fetch profile including avatar_url
                    profile = supabase.table("profiles").select("username, avatar_url").eq("id", res.user.id).execute()
                    username = "User"
                    avatar_url = None
                    if profile.data:
                        username = profile.data[0].get('username', 'User')
                        avatar_url = profile.data[0].get('avatar_url')
                    
                    return {
                        "logged_in": True, 
                        "email": res.user.email, 
                        "username": username,
                        "avatar_url": avatar_url
                    }
        except Exception as e:
            print(f"Session Restore Error: {e}")
            if os.path.exists(session_path): os.remove(session_path)
    return {"logged_in": False}

@eel.expose
def login(login_id, password):
    """Handles login with email OR username."""
    global current_user
    print(f"Login attempt for: {login_id}", flush=True)
    
    if not supabase: 
        print("Error: Supabase not initialized.", flush=True)
        return {"success": False, "error": "Supabase not initialized."}
    
    try:
        email = login_id
        if "@" not in login_id:
            print(f"Resolving username: {login_id}", flush=True)
            res = supabase.table("profiles").select("email").eq("username", login_id).execute()
            if res.data and res.data[0].get('email'):
                email = res.data[0]['email']
                print(f"Resolved to email: {email}", flush=True)
            else:
                return {"success": False, "error": "Username not found."}
        
        print("Authenticating with Supabase...", flush=True)
        auth_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        
        if auth_res and auth_res.user and auth_res.session:
            print(f"Auth success for user: {auth_res.user.id}", flush=True)
            current_user = {"id": auth_res.user.id, "email": auth_res.user.email}
            
            # Save session
            session_path = get_app_path("session.json")
            with open(session_path, "w") as f:
                json.dump({
                    "access_token": auth_res.session.access_token, 
                    "refresh_token": auth_res.session.refresh_token
                }, f)
            
            # Get username and avatar
            print("Fetching profile data...", flush=True)
            profile = supabase.table("profiles").select("username, avatar_url").eq("id", auth_res.user.id).execute()
            username = "User"
            avatar_url = None
            if profile.data:
                username = profile.data[0].get('username', 'User')
                avatar_url = profile.data[0].get('avatar_url')
            
            return {
                "success": True, 
                "user": {
                    "email": auth_res.user.email, 
                    "username": username,
                    "avatar_url": avatar_url
                }
            }
        else:
            print("Auth failed: No user/session in response.", flush=True)
            return {"success": False, "error": "Login failed."}

    except AuthApiError as e:
        print(f"Supabase Auth Error: {e.message}", flush=True)
        msg = e.message
        if "Invalid login credentials" in msg: msg = "Invalid email or password."
        return {"success": False, "error": msg}
    except Exception as e:
        print(f"Unexpected Login Error: {e}", flush=True)
        return {"success": False, "error": f"Internal Error: {str(e)}"}

@eel.expose
def register(username, email, password):
    """Handles user registration and profile creation."""
    global current_user
    if not supabase: return {"success": False, "error": "Supabase not initialized."}
    
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    try:
        # 1. Sign up user
        auth_res = supabase.auth.sign_up({"email": email, "password": password})
        if auth_res and auth_res.user:
            user_id = auth_res.user.id
            
            # 2. Create Profile
            supabase.table("profiles").insert({
                "id": user_id, 
                "username": username, 
                "email": email
            }).execute()
            
            # 3. Auto-login
            return login(email, password)
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def update_book_metadata(book_id, updates):
    """Updates book details in Supabase (Ownership checked)."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        # Verify ownership first
        check = supabase.table("books").select("owner_id").eq("id", book_id).single().execute()
        if not check.data or check.data["owner_id"] != current_user["id"]:
            return {"success": False, "error": "Unauthorized."}
            
        # Filter updates to only allowed fields
        allowed = ["title", "category", "is_public", "description", "tags", "is_read_only"]
        safe_updates = {k: v for k, v in updates.items() if k in allowed}
        
        res = supabase.table("books").update(safe_updates).eq("id", book_id).execute()
        return {"success": True}
    except Exception as e:
        print(f"Update Book Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def toggle_like(book_id):
    """Toggles a 'Pulse' (Like) on a book for the current user."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Login to pulse."}
    
    try:
        # In this prototype, we'll store likes as a count on the book itself
        # For a full system, we'd have a separate junction table
        book = supabase.table("books").select("likes").eq("id", book_id).single().execute()
        current_likes = book.data.get("likes", 0) or 0
        
        # Simple increment for prototype (no duplicate tracking for now)
        new_count = current_likes + 1
        supabase.table("books").update({"likes": new_count}).eq("id", book_id).execute()
        return {"success": True, "new_count": new_count}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_author_stats(user_id):
    """Fetches 'Neural Signature' stats for a specific author."""
    if not supabase: return {"success": False}
    try:
        # Get username/avatar from profile
        profile = supabase.table("profiles").select("username, avatar_url, bio").eq("id", user_id).single().execute()
        # Get count of books
        books = supabase.table("books").select("id, likes").eq("owner_id", user_id).execute()
        
        total_books = len(books.data) if books.data else 0
        total_likes = sum(b.get("likes", 0) or 0 for b in books.data) if books.data else 0
        
        return {
            "success": True,
            "username": profile.data.get("username", "Unknown"),
            "avatar_url": profile.data.get("avatar_url"),
            "bio": profile.data.get("bio", "Neural Architect"),
            "stats": {
                "contributions": total_books,
                "reputation": total_likes
            }
        }
    except Exception as e:
        print(f"Fetch Author Error: {e}")
        return {"success": False, "error": str(e)}

@eel.expose
def delete_book(book_id):
    """Deletes a book from Supabase (Ownership checked)."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
        
    try:
        # Verify ownership first
        check = supabase.table("books").select("owner_id").eq("id", book_id).single().execute()
        if not check.data or check.data["owner_id"] != current_user["id"]:
            return {"success": False, "error": "Unauthorized."}
            
        supabase.table("books").delete().eq("id", book_id).execute()
        return {"success": True}
    except Exception as e:
        print(f"Delete Book Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_user_projects():
    """Fetches books owned by the current user."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        res = supabase.table("books").select("*").eq("owner_id", current_user["id"]).order("created_at", desc=True).execute()
        return {"success": True, "projects": res.data}
    except Exception as e:
        print(f"Fetch Projects Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_global_library():
    """Fetches public books grouped by category."""
    if not supabase:
        return {"success": False, "error": "Supabase not initialized."}
    
    try:
        # Limit to 50 books for performance stability
        res = supabase.table("books").select("*").eq("is_public", True).order("created_at", desc=True).limit(50).execute()
        books = res.data
        
        # Group by Category (mimicking the original folder structure)
        grouped = {}
        for book in books:
            # Enhanced search data: mix title, category, and tags
            book['_search_index'] = f"{book.get('title','')} {book.get('category','')} {book.get('tags','')}".lower()
            
            cat = str(book.get('category', 'Uncategorized') or 'Uncategorized')
            if cat not in grouped: grouped[cat] = []
            grouped[cat].append(book)
            
        return {"success": True, "library": grouped}
    except Exception as e:
        print(f"Fetch Library Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def upload_avatar():
    """Opens a native file dialog, processes and uploads an avatar image."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}

    try:
        import tkinter as tk
        from tkinter import filedialog
        from PIL import Image
        import io
        import time

        # 1. Open hidden root for dialog (prevents empty window)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select Profile Picture",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp")]
        )
        root.destroy()

        if not file_path:
            return {"success": False, "cancelled": True}

        # 2. Process image
        img = Image.open(file_path)
        img.thumbnail((256, 256))
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        file_name = f"{current_user['id']}_{int(time.time())}.png"

        # 3. Upload to Supabase Storage
        print(f"Uploading avatar: {file_name}...", flush=True)
        supabase.storage.from_("avatars").upload(file_name, img_bytes, {"content-type": "image/png"})
        
        # 4. Get public URL
        public_url = supabase.storage.from_("avatars").get_public_url(file_name)
        
        # 5. Update DB
        supabase.table("profiles").update({"avatar_url": public_url}).eq("id", current_user["id"]).execute()
        
        print(f"Avatar updated: {public_url}", flush=True)
        return {"success": True, "avatar_url": public_url}

    except Exception as e:
        print(f"Upload Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def logout():
    """Clears session and logs out."""
    global current_user
    current_user = None
    session_path = get_app_path("session.json")
    if os.path.exists(session_path):
        os.remove(session_path)
    return {"success": True}


# ============================================================
# BOOK READER & PERMISSIONS API
# ============================================================

@eel.expose
def fetch_book_detail(book_id):
    """Fetches full book metadata including is_read_only and owner info."""
    if not supabase: return {"success": False, "error": "Supabase not initialized."}
    try:
        res = supabase.table("books").select("*").eq("id", book_id).single().execute()
        if res.data:
            book = res.data
            # Check if current user has edit permission
            can_edit = False
            is_owner = False
            
            if current_user:
                if book.get("owner_id") == current_user["id"]:
                    can_edit = True
                    is_owner = True
                elif not book.get("is_read_only"):
                    can_edit = True
                else:
                    # Check permission_requests
                    perm = supabase.table("permission_requests").select("status").eq("book_id", book_id).eq("requester_id", current_user["id"]).execute()
                    if perm.data and any(p["status"] == "accepted" for p in perm.data):
                        can_edit = True
            
            return {"success": True, "book": book, "can_edit": can_edit, "is_owner": is_owner}
        return {"success": False, "error": "Book not found."}
    except Exception as e:
        print(f"Fetch Book Detail Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_book_page(book_id, page_index):
    """Fetches a single page of a book by index."""
    if not supabase: return {"success": False, "error": "Supabase not initialized."}
    try:
        # Get total pages
        book = supabase.table("books").select("total_pages").eq("id", book_id).single().execute()
        total = book.data.get("total_pages", 0) if book.data else 0
        
        # Get the specific page
        page = supabase.table("pages").select("*").eq("book_id", book_id).eq("page_index", page_index).single().execute()
        
        if page.data:
            # Generate public URL for the image
            img_url = supabase.storage.from_("book-images").get_public_url(f"{book_id}/img_{page_index}.jpg")
            page.data['image_url'] = img_url
            return {"success": True, "page": page.data, "total_pages": total}
        return {"success": True, "page": None, "total_pages": total}
    except Exception as e:
        print(f"Fetch Page Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def clone_book(book_id):
    """Clones a public book and all its pages to the current user's projects."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        # 1. Get original book
        orig = supabase.table("books").select("*").eq("id", book_id).single().execute()
        if not orig.data:
            return {"success": False, "error": "Book not found."}
        
        ob = orig.data
        
        # 2. Get profile for username
        profile = supabase.table("profiles").select("username").eq("id", current_user["id"]).single().execute()
        username = profile.data.get("username", "User") if profile.data else "User"
        
        # 3. Create new book
        new_book = supabase.table("books").insert({
            "title": f"{ob['title']} (Copy)",
            "total_pages": ob.get("total_pages", 0),
            "language": ob.get("language", ""),
            "category": ob.get("category", "Uncategorized"),
            "is_public": False,
            "is_read_only": False,
            "owner_id": current_user["id"],
            "owner_username": username,
            "likes": 0
        }).execute()
        
        if not new_book.data:
            return {"success": False, "error": "Failed to create book copy."}
        
        new_id = new_book.data[0]["id"]
        
        # 4. Copy all pages
        pages = supabase.table("pages").select("*").eq("book_id", book_id).order("page_index").execute()
        if pages.data:
            for p in pages.data:
                supabase.table("pages").insert({
                    "book_id": new_id,
                    "page_index": p["page_index"],
                    "original": p.get("original"),
                    "english": p.get("english"),
                    "literal": p.get("literal"),
                    "is_image": p.get("is_image", False),
                    "is_cover": p.get("is_cover", False),
                    "is_centered": p.get("is_centered", False),
                    "is_rtl_page": p.get("is_rtl_page", False)
                }).execute()
        
        print(f"Cloned book {book_id} -> {new_id}", flush=True)
        return {"success": True, "new_book_id": new_id}
    except Exception as e:
        print(f"Clone Book Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def request_edit_access(book_id):
    """Sends an edit access request to the book's owner."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        # Get requester username
        profile = supabase.table("profiles").select("username").eq("id", current_user["id"]).single().execute()
        username = profile.data.get("username", "User") if profile.data else "User"
        
        # Insert request (UNIQUE constraint prevents duplicates)
        supabase.table("permission_requests").insert({
            "book_id": book_id,
            "requester_id": current_user["id"],
            "requester_username": username,
            "status": "pending"
        }).execute()
        
        print(f"Edit request sent for book {book_id} by {username}", flush=True)
        return {"success": True}
    except Exception as e:
        err = str(e)
        if "duplicate" in err.lower() or "unique" in err.lower():
            return {"success": False, "error": "You have already requested access to this book."}
        print(f"Request Access Error: {e}", flush=True)
        return {"success": False, "error": err}

@eel.expose
def submit_suggestion(book_id, page_index, original_text, suggested_text):
    """Submits a translation suggestion as a comment."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        profile = supabase.table("profiles").select("username").eq("id", current_user["id"]).single().execute()
        username = profile.data.get("username", "User") if profile.data else "User"
        
        content = f"[SUGGESTION] Page {page_index + 1}\n---\nOriginal: {original_text}\n---\nSuggested: {suggested_text}"
        
        supabase.table("comments").insert({
            "book_id": book_id,
            "user_id": current_user["id"],
            "username": username,
            "content": content
        }).execute()
        
        return {"success": True}
    except Exception as e:
        print(f"Submit Suggestion Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_inbox_requests():
    """Fetches pending edit requests for books owned by the current user."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        # Get all books owned by this user
        books = supabase.table("books").select("id, title").eq("owner_id", current_user["id"]).execute()
        if not books.data:
            return {"success": True, "requests": []}
        
        book_ids = [b["id"] for b in books.data]
        book_map = {b["id"]: b["title"] for b in books.data}
        
        # Get pending requests for those books
        requests = supabase.table("permission_requests").select("*").in_("book_id", book_ids).eq("status", "pending").order("created_at", desc=True).execute()
        
        # Enrich with book titles
        result = []
        for req in (requests.data or []):
            req["book_title"] = book_map.get(req["book_id"], "Unknown")
            result.append(req)
        
        return {"success": True, "requests": result}
    except Exception as e:
        print(f"Fetch Inbox Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def respond_to_request(request_id, accept):
    """Accepts or denies an edit access request."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    try:
        new_status = "accepted" if accept else "denied"
        supabase.table("permission_requests").update({"status": new_status}).eq("id", request_id).execute()
        
        print(f"Request {request_id} -> {new_status}", flush=True)
        return {"success": True, "status": new_status}
    except Exception as e:
        print(f"Respond Request Error: {e}", flush=True)
        return {"success": False, "error": str(e)}


# ============================================================
# MAIN — Launch the desktop window
# ============================================================

def main():
    try:
        # Try to open in Edge (App Mode — no address bar, looks native)
        # We use cmdline_args to enforce physical window limits in Chromium
        eel.start(
            "index.html",
            mode="edge",
            size=(1450, 900),
            cmdline_args=['--window-size=1450,900', '--min-window-size=1024,600'],
            port=8087,          # Fixed port for stability
            block=True,          # Block until window is closed
        )
    except EnvironmentError:
        # Fallback: try Chrome
        try:
            eel.start(
                "index.html", 
                mode="chrome", 
                size=(1450, 900), 
                cmdline_args=['--window-size=1450,900', '--min-window-size=1024,600'],
                port=8080, 
                block=True
            )
        except EnvironmentError:
            # Last resort: open in default browser
            eel.start("index.html", mode="default", size=(1450, 900), port=8080, block=True)


# ============================================================
# TRANSLATOR CORE API
# ============================================================

@eel.expose
def get_languages():
    """Returns the list of supported languages."""
    return {"success": True, "languages": list(LANGUAGES.keys())}

@eel.expose
def trigger_pdf_upload():
    """Opens a native file dialog and loads the selected PDF."""
    # We need a hidden Tk root for the file dialog
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select a PDF to translate",
        filetypes=[("PDF Files", "*.pdf")],
        parent=root
    )
    root.destroy()
    
    if not file_path:
        return {"success": False, "error": "No file selected."}
    
    result = translator.load_pdf(file_path)
    return result

@eel.expose
def start_translation(source_lang="Auto-Detect", target_lang="English", engine="Google"):
    """Starts the OCR + Translation pipeline."""
    return translator.start_translation(source_lang, target_lang, engine)

@eel.expose
def rescan_translator_page(page_index, source_lang="Auto-Detect", target_lang="English", engine="Google"):
    """Forces a single page to be re-scanned and translated."""
    return translator.start_translation(source_lang, target_lang, engine, pages_to_process=[page_index])

@eel.expose
def stop_translation():
    """Stops the running translation pipeline."""
    translator.stop()
    return {"success": True}

@eel.expose
def clear_translation():
    """Fully resets the translator state."""
    translator.clear()
    return {"success": True}

@eel.expose
def load_translator_page(page_index):
    """Loads a specific page's data (image, OCR text, translation)."""
    return translator.get_page_data(page_index)

@eel.expose
def get_page_image_base64(page_index):
    """Returns the cached page image as a base64 data URL for rendering in the browser."""
    import base64
    img_path = translator.get_page_image_path(page_index)
    if img_path and os.path.exists(img_path.replace("/", os.sep)):
        try:
            with open(img_path.replace("/", os.sep), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                return {"success": True, "data_url": f"data:image/jpeg;base64,{b64}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "Image not found"}

@eel.expose
def save_translator_page_edits(page_index, ocr_text, trans_text):
    """Saves user manual edits to the current page's OCR and translated text."""
    return translator.save_page_edits(page_index, ocr_text, trans_text)

@eel.expose
def quick_translate_text(text, source_lang, target_lang, engine):
    """Performs a quick standalone translation."""
    return translator.quick_translate_text(text, source_lang, target_lang, engine)

@eel.expose
def quick_translate_image(base64_data, source_lang, target_lang, engine):
    """Performs a quick standalone OCR and translation from an image."""
    return translator.quick_translate_image(base64_data, source_lang, target_lang, engine)

@eel.expose
def search_pdf_text(query):
    """Searches through all cached pages for matching text."""
    return translator.search_text(query)

@eel.expose
def get_local_projects():
    """Scans the local metadata cache to return a list of in-progress sessions for the UI."""
    from utils import get_app_path
    import json
    import time
    
    cache_base = get_app_path(".translator_cache")
    if not os.path.exists(cache_base):
        return {"success": True, "projects": []}
        
    projects = []
    for d in os.listdir(cache_base):
        dir_path = os.path.join(cache_base, d)
        info_path = os.path.join(dir_path, "project_info.json")
        if os.path.isdir(dir_path) and os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Fetch cover image base64 if it exists
                # We can just return the path or let the UI fetch it.
                # Since image fetching depends on 'page_index' for the loaded book, 
                # we can return a quick file path or just rely on a default cover for now.
                img_path = os.path.join(dir_path, "img_0.jpg")
                cover_base64 = None
                if os.path.exists(img_path):
                    import base64
                    with open(img_path, "rb") as img_file:
                        code = base64.b64encode(img_file.read()).decode('utf-8')
                        cover_base64 = f"data:image/jpeg;base64,{code}"

                # Count actual completed pages
                completed_pages = 0
                for cache_file in os.listdir(dir_path):
                    if cache_file.startswith("page_") and cache_file.endswith(".json"):
                        completed_pages += 1
                        
                meta["completed_pages"] = completed_pages
                meta["cover_base64"] = cover_base64
                meta["id"] = d # use folder name as unique ID
                projects.append(meta)
            except Exception as e:
                print(f"Error reading project {d}: {e}")
                
    # Sort by last_accessed descending
    projects.sort(key=lambda x: x.get("last_accessed", 0), reverse=True)
    return {"success": True, "projects": projects}

@eel.expose
def resume_local_project(cache_hash):
    """Resumes an existing local translation project from the cache folder."""
    from utils import get_app_path
    
    dir_path = get_app_path(os.path.join(".translator_cache", cache_hash))
    source_pdf = os.path.join(dir_path, "source.pdf")
    
    if not os.path.exists(source_pdf):
        return {"success": False, "error": "Original PDF source missing from cache."}
        
    # Standard load_pdf flow 
    res = translator.load_pdf(source_pdf)
    if res.get("success"):
        # We need to jump the UI to the last accessed page.
        # We also need to read the project_info.json to know that.
        info_path = os.path.join(dir_path, "project_info.json")
        last_page = 0
        src_lang = "Auto-Detect"
        tgt_lang = "English"
        if os.path.exists(info_path):
            import json
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                last_page = meta.get("last_translated_page", 0)
                src_lang = meta.get("source_lang", "Auto-Detect")
                tgt_lang = meta.get("target_lang", "English")
            except:
                pass
                
        res["last_page"] = last_page
        res["source_lang"] = src_lang
        res["target_lang"] = tgt_lang
        
    return res

@eel.expose
def delete_local_project(cache_hash):
    """Deletes an in-progress local translation project completely."""
    import shutil
    from utils import get_app_path
    
    dir_path = get_app_path(os.path.join(".translator_cache", cache_hash))
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "Project not found."}

@eel.expose
def publish_book_to_cloud(title, category, description, tags, is_public, is_read_only, source_lang, target_lang):
    """Publishes the current translation project to Supabase."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    if not translator.pdf_path or not translator.all_page_data:
        return {"success": False, "error": "No translation loaded."}
    
    try:
        # 1. Create book record
        book_data = {
            "title": title,
            "owner_id": current_user["id"],
            "category": category.upper() if category else "GENERAL",
            "description": description,
            "tags": tags.upper() if tags else "",
            "is_public": is_public,
            "is_read_only": is_read_only,
            "language": f"{source_lang} → {target_lang}",
            "total_pages": translator.total_pages,
        }
        res = supabase.table("books").insert(book_data).execute()
        if not res.data:
            return {"success": False, "error": "Failed to create book record."}
        
        book_id = res.data[0]["id"]
        
        # 2. Upload pages
        for i, page_data in enumerate(translator.all_page_data):
            if page_data is None:
                continue
            page_record = {
                "book_id": book_id,
                "page_index": i,
                "original": page_data.get("original", ""),
                "english": page_data.get("english", ""),
                "literal": page_data.get("literal", ""),
            }
            supabase.table("pages").insert(page_record).execute()
            
            # 3. Upload page image to storage
            img_path = os.path.join(translator.cache_dir, f"img_{i}.jpg")
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        supabase.storage.from_("book-images").upload(
                            f"{book_id}/img_{i}.jpg",
                            f.read(),
                            {"content-type": "image/jpeg"}
                        )
                except Exception as upload_err:
                    print(f"Image upload failed for page {i}: {upload_err}", flush=True)
        
        return {"success": True, "book_id": book_id}
    except Exception as e:
        print(f"Publish Error: {e}", flush=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    main()
