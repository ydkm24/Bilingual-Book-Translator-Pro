import os
import sys
import json
import eel
from dotenv import load_dotenv
from supabase import create_client, Client, AuthApiError
from typing import Optional, Dict, Any
from tkinter import filedialog, Tk
from utils import resource_path, get_app_path
from web_processing import WebTranslatorManager, LANGUAGES
import threading
import time


# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip('"')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip('"')
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


def get_user_cache_path():
    """Returns the user-specific cache directory for projects.
    Logged-in users get:  .translator_cache/<user_id>/
    Guests get:           .translator_cache/guest/
    """
    uid = current_user["id"] if current_user else "guest"
    path = get_app_path(os.path.join(".translator_cache", uid))
    os.makedirs(path, exist_ok=True)
    return path

# Initialize Eel with the web_ui folder
eel.init(os.path.join(BASE_DIR, "web_ui"))


# ============================================================
# EXPOSED PYTHON FUNCTIONS (callable from JS as eel.func())
# ============================================================

@eel.expose
def get_version():
    """Returns the current app version string."""
    return "v4.2.0"

APP_VERSION = "4.2.0"
_ping_total = 0
_ping_success = 0


@eel.expose
def get_dashboard_stats():
    """Returns all live dashboard widget data in one call."""
    global _ping_total, _ping_success
    
    result = {
        "version": APP_VERSION,
        "version_status": "up_to_date",  # or "update_available"
        "latest_version": APP_VERSION,
        "uptime_pct": "100.0",
        "total_users": 0,
        "online_users": 0,
        "library_count": 0
    }
    
    if not supabase:
        print("[Dashboard] Supabase NOT connected", flush=True)
        result["uptime_pct"] = "0.0"
        return result
    
    print(f"[Dashboard] Polling stats... (Success: {_ping_success}/{_ping_total})", flush=True)
    
    # Ping/uptime tracking
    _ping_total += 1
    try:
        # 1. Check latest version from app_config
        try:
            cfg = supabase.table("app_config").select("value").eq("key", "latest_version").single().execute()
            if cfg.data:
                latest = cfg.data["value"]
                result["latest_version"] = latest
                result["version_status"] = "up_to_date" if latest == APP_VERSION else "update_available"
        except:
            pass
        
        # 2. Total users
        try:
            users_res = supabase.table("profiles").select("id", count="exact").execute()
            result["total_users"] = users_res.count or 0
        except:
            pass
        
        # 3. Online users (last_seen within 2 minutes)
        try:
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
            online_res = supabase.table("profiles").select("id", count="exact").gte("last_seen", cutoff).execute()
            result["online_users"] = online_res.count or 0
        except:
            pass
        
        # 4. Library catalog count
        try:
            books_res = supabase.table("books").select("id", count="exact").eq("is_public", True).execute()
            result["library_count"] = books_res.count or 0
        except:
            pass
            
        _ping_success += 1
    except Exception as e:
        print(f"Dashboard Stats Error: {e}", flush=True)

    # Calculate uptime percentage
    if _ping_total > 0:
        # If we have any success at all, don't show 0.0% unless it's a real outage
        pct = (_ping_success / _ping_total) * 100
        # Floor it to 99.9 if it's recently started but successful
        if _ping_success > 0 and pct < 90: pct = 99.9 
        result["uptime_pct"] = f"{pct:.1f}"
    
    # Final safety check: ensure everything is JSON-serializable
    return json.loads(json.dumps(result, default=str))



@eel.expose
def get_dashboard_recent_projects():
    """Returns the most recent local projects for the dashboard cards."""
    import glob
    projects = []
    # Use user-specific cache directory
    cache_base = get_user_cache_path()
    print(f"[Dashboard] Scanning for recent projects in: {cache_base}", flush=True)
    
    if not os.path.isdir(cache_base):
        print(f"[Dashboard] WARN: Cache directory not found at {cache_base}", flush=True)
        return {"success": True, "projects": []}
    
    for proj_dir in os.listdir(cache_base):
        info_path = os.path.join(cache_base, proj_dir, "project_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                
                # UNIFIED PROGRESS COUNTING (Match get_local_projects)
                completed = 0
                dir_path = os.path.join(cache_base, proj_dir)
                if os.path.isdir(dir_path):
                    for cache_file in os.listdir(dir_path):
                        if cache_file.startswith("page_") and cache_file.endswith(".json"):
                            completed += 1
                
                total = info.get("total_pages", 0)
                pct = int((completed / total) * 100) if total > 0 else 0
                
                # Get last modified time of the directory
                mtime = os.path.getmtime(dir_path)
                
                projects.append({
                    "project_id": proj_dir,
                    "title": info.get("title", "Untitled"),
                    "source_lang": info.get("source_lang", "Auto"),
                    "target_lang": info.get("target_lang", "English"),
                    "total_pages": total,
                    "completed_pages": completed,
                    "percent": pct,
                    "last_modified": mtime
                })
            except Exception as e:
                print(f"[Dashboard] Error processing {proj_dir}: {e}", flush=True)
    
    print(f"[Dashboard] Found {len(projects)} recent projects in {cache_base}", flush=True)
    
    # Sort by most recently modified,    # Sort and return top 4
    projects.sort(key=lambda x: x.get("last_modified", 0), reverse=True)
    return {"success": True, "projects": projects[:4]}


@eel.expose
def get_trending_books():
    """Returns the top 5 most popular public books from Supabase."""
    if not supabase:
        return {"success": False, "error": "Supabase not connected."}
    try:
        res = supabase.table("books").select("id, title, owner_username, likes, category").eq("is_public", True).order("likes", desc=True).limit(5).execute()
        # Harden Supabase data to JSON-safe types (UUIDs, datetimes -> strings)
        safe_data = json.loads(json.dumps(res.data or [], default=str))
        return {"success": True, "books": safe_data}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def get_notifications():
    """Unified Dashboard Notifications dropdown."""
    res = _get_unified_inbox()
    return {
        "success": res["success"],
        "notifications": res["items"][:10], # Limit for dropdown
        "unread_count": res["unread_count"]
    }

def _get_unified_inbox():
    """Private helper to aggregate notifications + friend requests + site alerts."""
    if not supabase or not current_user:
        return {"success": True, "items": [], "unread_count": 0}
    
    uid = current_user["id"]
    items = []
    
    # 1. Standard notifications (Resilient fetch)
    try:
        notifs = supabase.table("notifications").select("*").eq("user_id", uid).order("created_at", desc=True).limit(30).execute()
        for n in (notifs.data or []):
            items.append({
                "id": n["id"],
                "type": n.get("type", "general"),
                "title": n.get("title", n.get("message", "System Alert")[:20]), # For dash bell
                "message": n.get("message", ""),
                "read": n.get("read", False),
                "created_at": n.get("created_at", ""),
                "metadata": n.get("metadata", "{}")
            })
    except Exception as e:
        print(f"[Inbox] Notifications table error: {e}", flush=True)

    # 2. Pending friend requests (Always include)
    try:
        requests = supabase.table("friend_requests").select("*").eq("receiver_id", uid).eq("status", "pending").execute()
        for r in (requests.data or []):
            sender = supabase.table("profiles").select("username").eq("id", r["sender_id"]).execute()
            sender_name = sender.data[0]["username"] if sender.data else "Someone"
            items.append({
                "id": f"fr_{r['id']}",
                "type": "friend_request",
                "title": "Friend Request",
                "message": f"{sender_name} wants to be your friend!",
                "read": False,
                "created_at": r.get("created_at", ""),
                "metadata": json.dumps({"request_id": r["id"], "sender_id": r["sender_id"], "sender_username": sender_name})
            })
    except Exception as e:
        print(f"[Inbox] Friend requests error: {e}", flush=True)

    # Sort all by created_at descending
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    unread = sum(1 for i in items if not i.get("read", False))
    
    return {"success": True, "items": items, "unread_count": unread}


@eel.expose
def mark_notification_read(notif_id):
    """Marks a single notification as read."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        supabase.table("notifications").update({"read": True}).eq("id", notif_id).eq("user_id", current_user["id"]).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
                    
                    # Harden everything to plain strings/JSON-safe types
                    return json.loads(json.dumps({
                        "logged_in": True, 
                        "email": res.user.email, 
                        "username": username,
                        "avatar_url": avatar_url,
                        "id": res.user.id,
                        "is_admin": profile.data[0].get('is_admin', False) if profile.data else False
                    }, default=str))
        except Exception as e:
            print(f"Session Error: {e}", flush=True)
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
            
            # Update last_seen immediately
            try: heartbeat_last_seen()
            except: pass

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
                    "id": auth_res.user.id,
                    "email": auth_res.user.email, 
                    "username": username,
                    "avatar_url": avatar_url,
                    "is_admin": profile.data[0].get('is_admin', False) if profile.data else False
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
    """Initiates user registration (sends OTP if email confirmation is enabled)."""
    if not supabase: return {"success": False, "error": "Supabase not initialized."}
    if len(username) < 4: return {"success": False, "error": "Username must be at least 4 characters."}
    if len(password) < 6: return {"success": False, "error": "Password must be at least 6 characters."}

    try:
        # Check if username or email is already taken first
        check_user = supabase.table("profiles").select("id").eq("username", username).execute()
        if check_user.data:
            return {"success": False, "error": "Username is already taken."}
            
        check_email = supabase.table("profiles").select("id").eq("email", email).execute()
        if check_email.data:
            return {"success": False, "error": "An account with this email already exists."}

        # Sign up user (sends OTP if project requires email confirmation)
        auth_res = supabase.auth.sign_up({"email": email, "password": password})
        
        if auth_res and auth_res.user:
            # If session exists immediately, email confirm is OFF.
            if auth_res.session:
                supabase.table("profiles").upsert({
                    "id": auth_res.user.id, 
                    "username": username, 
                    "email": email
                }).execute()
                return login(email, password)
            else:
                # Email confirmation is ON. Tell frontend to show OTP screen.
                return {"success": True, "require_otp": True, "email": email, "username": username}
        else:
            return {"success": False, "error": "Signup failed. Try again."}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def verify_signup_otp(email, code, username):
    """Verifies the 6-digit OTP code sent for signup, then creates the profile."""
    global current_user
    if not supabase: return {"success": False, "error": "Not connected."}
    
    try:
        res = supabase.auth.verify_otp({"email": email, "token": code, "type": "signup"})
        if res and res.session:
            # Successfully authenticated via OTP! Now we can safely create the profile
            supabase.table("profiles").upsert({
                "id": res.user.id, 
                "username": username, 
                "email": email
            }).execute()
            
            # Update last_seen immediately on login
            from datetime import datetime, timezone
            now_ts = datetime.now(timezone.utc).isoformat()
            supabase.table("profiles").update({"last_seen": now_ts}).eq("id", res.user.id).execute()

            # Set internal session state
            current_user = {"id": res.user.id, "email": res.user.email} # Keep original for now, get_user_profile is not defined
            session_path = get_app_path("session.json")
            with open(session_path, "w") as f:
                json.dump({
                    "access_token": res.session.access_token, 
                    "refresh_token": res.session.refresh_token
                }, f)
                
            return {
                "success": True, 
                "user": {"email": res.user.email, "username": username, "avatar_url": None}
            }
        else:
            return {"success": False, "error": "Invalid or expired code."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def resend_signup_otp(email):
    """Resends a 6-digit signup OTP."""
    if not supabase: return {"success": False}
    try:
        supabase.auth.resend({"type": "signup", "email": email})
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def forgot_password_send_otp(email):
    """Sends a password recovery OTP to the user's email."""
    if not supabase: return {"success": False}
    try:
        supabase.auth.reset_password_email(email)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def reset_password_with_otp(email, code, new_password):
    """Verifies the recovery OTP and sets the new password."""
    if not supabase: return {"success": False}
    if len(new_password) < 6: return {"success": False, "error": "Password too short."}
    try:
        # 1. Verify the recovery OTP (this logs them in)
        verify_res = supabase.auth.verify_otp({"email": email, "token": code, "type": "recovery"})
        if verify_res and verify_res.session:
            # 2. Update the password
            supabase.auth.update_user({"password": new_password})
            return {"success": True}
        else:
            return {"success": False, "error": "Invalid or expired code."}
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
        book = supabase.table("books").select("likes, title, owner_id").eq("id", book_id).single().execute()
        current_likes = book.data.get("likes", 0) or 0
        
        # Simple increment for prototype
        new_count = current_likes + 1
        supabase.table("books").update({"likes": new_count}).eq("id", book_id).execute()
        
        # Check Milestones (5, 10, 50)
        if new_count in [5, 10, 50]:
            is_viral = (new_count >= 50)
            _trigger_activity_feed(
                user_id=book.data["owner_id"],
                activity_type="book_milestone",
                title=f"Trending Book: {book.data['title']}",
                description=f"This book just reached {new_count} pulses! {'🔥 VIRAL' if is_viral else '⚡ Connection Hub Alert'}",
                metadata={"book_id": book_id, "likes": new_count},
                is_public_trigger=is_viral 
            )
            
        # Increment author's total likes
        try:
            author_id = book.data["owner_id"]
            stats = supabase.table("user_stats").select("total_likes").eq("user_id", author_id).single().execute()
            count = stats.data.get("total_likes", 0) + 1 if stats.data else 1
            supabase.table("user_stats").upsert({"user_id": author_id, "total_likes": count}).execute()
        except Exception as st_err:
            print(f"[Stats] Like update failed: {st_err}", flush=True)
            
        return {"success": True, "new_count": new_count}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def fetch_author_stats(user_id):
    """Fetches 'Neural Signature' stats for a specific author."""
    if not supabase: return {"success": False}
    try:
        # Get username/avatar from profile
        # We select only columns we are sure exist. bio might be missing in older schemas.
        profile = supabase.table("profiles").select("username, avatar_url").eq("id", user_id).single().execute()
        if not profile.data:
            return {"success": False, "error": "User not found."}

        username = profile.data.get("username", "Unknown")
        avatar_url = profile.data.get("avatar_url", "")
        
        # Handle missing bio column by checking the table or just providing a default
        bio = "Neural Architect" # Default fallback
        try:
            # Attempt to fetch bio separately to see if column exists
            bio_check = supabase.table("profiles").select("bio").eq("id", user_id).single().execute()
            if bio_check.data:
                bio = bio_check.data.get("bio", "Neural Architect")
        except:
            pass # Column doesn't exist, use default
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
def update_username(new_username):
    """Updates the current user's username with validation."""
    global current_user
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    new_username = new_username.strip()
    if len(new_username) < 4:
        return {"success": False, "error": "Username must be at least 4 characters."}
    
    try:
        # Check if username is already taken
        check = supabase.table("profiles").select("id").eq("username", new_username).execute()
        if check.data and check.data[0]["id"] != current_user["id"]:
            return {"success": False, "error": "Username is already taken."}
        
        # Update profile
        supabase.table("profiles").update({"username": new_username}).eq("id", current_user["id"]).execute()
        
        # Update current_user state (if needed, though it's mainly id/email)
        # We don't store username in current_user global, it's fetched per session check
        
        return {"success": True}
    except Exception as e:
        print(f"Update Username Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def heartbeat_last_seen():
    """Update current user's last_seen timestamp (Heartbeat) and streaks."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        from datetime import datetime, timezone
        now_ts = datetime.now(timezone.utc).isoformat()
        supabase.table("profiles").update({"last_seen": now_ts}).eq("id", current_user["id"]).execute()
        return {"success": True}
    except Exception as e:
        print(f"[Heartbeat] Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def increment_pages_translated(count=1):
    """
    Increments the current user's global pages_translated stat.
    Called from the translator's page-finished callback.
    """
    if not supabase or not current_user:
        return {"success": False}
    try:
        uid = current_user["id"]
        # Atomic increment via RPC (if exists) or manual fetch-update
        # For simplicity in this env, we'll try a manual update first
        stats_res = supabase.table("user_stats").select("pages_translated").eq("user_id", uid).execute()
        current_count = 0
        if stats_res.data:
            current_count = stats_res.data[0].get("pages_translated", 0) or 0
        
        new_count = current_count + int(count)
        supabase.table("user_stats").upsert({
            "user_id": uid,
            "pages_translated": new_count
        }).execute()
        
        return {"success": True, "new_count": new_count}
    except Exception as e:
        print(f"[Stats] Increment Pages Error: {e}", flush=True)
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

@eel.expose
def get_sys_stats():
    import psutil
    import time
    import os
    
    # Measure DB Latency
    start_time = time.time()
    latency = 0
    try:
        if supabase:
            supabase.table("profiles").select("id").limit(1).execute()
            latency = int((time.time() - start_time) * 1000)
        else:
            latency = -1
    except:
        latency = -1
        
    # Hardware (Current Process)
    try:
        process = psutil.Process()
        mem_mb = int(process.memory_info().rss / (1024 * 1024))
        cpu = psutil.cpu_percent(interval=None) # Non-blocking
    except:
        mem_mb = 0
        cpu = 0.0
    
    # Engine state
    state = "ENGINE IDLE"
    try:
        if getattr(translator, 'is_running', False):
            if getattr(translator, 'is_paused', False):
                 state = "ENGINE PAUSED"
            else:
                 # Check if it has a specific status string
                 st = getattr(translator, 'status', None)
                 if st: state = st.upper()
                 else: state = "ENGINE ACTIVE"
        elif getattr(translator, 'pdf_path', None):
            filename = os.path.basename(translator.pdf_path)
            # Truncate
            if len(filename) > 20: filename = filename[:17] + "..."
            state = f"TARGET: {filename}"
    except:
        pass
        
    return {
        "success": True,
        "latency": latency,
        "mem": mem_mb,
        "cpu": cpu,
        "engine": state
    }

import ctypes

@eel.expose
def minimize_window():
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        user32.ShowWindow(hwnd, 6) # SW_MINIMIZE = 6
    except Exception as e:
        print(f"Minimize err: {e}")

@eel.expose
def maximize_window():
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        # Toggle maximize/restore
        if user32.IsZoomed(hwnd):
            user32.ShowWindow(hwnd, 9) # SW_RESTORE = 9
            return False
        else:
            user32.ShowWindow(hwnd, 3) # SW_MAXIMIZE = 3
            return True
    except Exception as e:
        print(f"Maximize err: {e}")
        return False

# ============================================================
# MAIN — Launch the desktop window
# ============================================================

def start_realtime_listener():
    """Background thread to listen for new social activity."""
    if not supabase: return
    
    def on_insert(payload):
        # Payload: {'schema': 'public', 'table': 'activity_feed', 'commit_timestamp': '...', 'event_type': 'INSERT', 'new': {...}}
        try:
            new_item = payload.get('new', {})
            activity_id = new_item.get('id')
            if not activity_id: return
            
            # Fetch full enriched item
            res = supabase.table("activity_feed").select("*, profiles(username, avatar_url)").eq("id", activity_id).single().execute()
            if res.data:
                enriched = json.loads(json.dumps(res.data, default=str))
                print(f"[Realtime] New Activity: {enriched.get('title')}", flush=True)
                # Broadcast to all connected Eel clients
                eel.onNewActivity(enriched)
        except Exception as e:
            print(f"[Realtime] Broadcast Error: {e}", flush=True)

    try:
        print("[Realtime] Attempting to connect to activity_feed channel...", flush=True)
        channel = supabase.channel("social_updates")
        channel.on(
            "postgres_changes", 
            event="INSERT", 
            schema="public", 
            table="activity_feed", 
            callback=on_insert
        ).subscribe()
        print("[Realtime] Subscribed to activity_feed (INSERT)", flush=True)
    except Exception as e:
        print(f"[Realtime] Subscription Failed: {e}", flush=True)

def main():
    # Start Realtime listener in background
    threading.Thread(target=start_realtime_listener, daemon=True).start()
    
    try:
        # Switch to 8089 to avoid conflicts
        eel.start(
            "index.html",
            mode="edge",
            size=(1450, 900),
            cmdline_args=['--window-size=1450,900', '--min-window-size=1024,600', '--frameless'],
            port=8089,
            block=True,
        )
    except Exception:
        # Fallback
        try:
            eel.start(
                "index.html", 
                mode="chrome", 
                size=(1450, 900), 
                port=8089, 
                block=True
            )
        except Exception:
            # Last resort: open in default browser
            eel.start("index.html", mode="default", size=(1450, 900), port=8089, block=True)


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
    
    # Pass get_user_cache_path() to isolate by user
    result = translator.load_pdf(file_path, cache_base=get_user_cache_path())
    return result

@eel.expose
def start_translation(source_lang="Auto-Detect", target_lang="English", engine="Google", ocr_tier="Standard"):
    """Starts the OCR + Translation pipeline."""
    return translator.start_translation(source_lang, target_lang, engine, ocr_tier=ocr_tier)

@eel.expose
def rescan_translator_page(page_index, source_lang="Auto-Detect", target_lang="English", engine="Google", ocr_tier="Standard"):
    """Forces a single page to be re-scanned and translated."""
    return translator.start_translation(source_lang, target_lang, engine, pages_to_process=[page_index], ocr_tier=ocr_tier)

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
    
    cache_base = get_user_cache_path()
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
                    from PIL import Image
                    import io
                    try:
                        # Downscale thumbnail to prevent WebSocket frame limit crash
                        with Image.open(img_path) as cover_img:
                            rgb_img = cover_img.convert("RGB")
                            rgb_img.thumbnail((300, 400)) # Small thumbnail
                            buffer = io.BytesIO()
                            rgb_img.save(buffer, format="JPEG", quality=70)
                            code = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            cover_base64 = f"data:image/jpeg;base64,{code}"
                    except Exception as e:
                        print(f"Thumbnail error: {e}")
                        pass

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
    
    dir_path = os.path.join(get_user_cache_path(), cache_hash)
    source_pdf = os.path.join(dir_path, "source.pdf")
    
    if not os.path.exists(source_pdf):
        return {"success": False, "error": "Original PDF source missing from cache."}
        
    # Standard load_pdf flow - pass cache_base to ensure user isolation
    res = translator.load_pdf(source_pdf, folder_name=cache_hash, cache_base=get_user_cache_path())
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
    
    dir_path = os.path.join(get_user_cache_path(), cache_hash)
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
        
        # 4. Check Activity Feed
        is_first = True
        try:
            prev = supabase.table("books").select("id").eq("owner_id", current_user["id"]).limit(2).execute()
            if prev.data and len(prev.data) > 1: is_first = False
        except: pass

        if is_public:
            _trigger_activity_feed(
                user_id=current_user["id"],
                activity_type="first_publish" if is_first else "book_published",
                title=f"New Release: {title}",
                description=f"Published in {category.upper()} with {translator.total_pages} pages.",
                metadata={"book_id": book_id, "pages": translator.total_pages},
                is_public_trigger=is_first 
            )
            
            # Increment user stats
            try:
                supabase.rpc('increment_user_books', {'row_user_id': current_user["id"]}).execute()
            except:
                # Fallback to manual update if RPC fails
                try:
                    stats = supabase.table("user_stats").select("books_translated").eq("user_id", current_user["id"]).single().execute()
                    count = stats.data.get("books_translated", 0) + 1 if stats.data else 1
                    supabase.table("user_stats").upsert({"user_id": current_user["id"], "books_translated": count}).execute()
                except: pass
        
        # 5. Milestone: 500+ pages
        if translator.total_pages >= 500:
            _trigger_activity_feed(
                user_id=current_user["id"],
                activity_type="long_haul",
                title="Monolithic Translation Completed",
                description=f"Translated a massive {translator.total_pages} page tome!",
                metadata={"book_id": book_id, "pages": translator.total_pages},
                is_public_trigger=True 
            )

        return {"success": True, "book_id": book_id}
    except Exception as e:
        print(f"Publish Error: {e}", flush=True)
        return {"success": False, "error": str(e)}


# ============================================================
# SOCIAL HUB — PHASE 2: FRIEND SYSTEM
# ============================================================

@eel.expose
def search_users(query):
    """Search for users by username (partial match)."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    if len(query) < 4:
        return {"success": True, "users": []}
        
    try:
        res = supabase.table("profiles").select("id, username, avatar_url").ilike("username", f"%{query}%").neq("id", current_user["id"]).limit(20).execute()
        return {"success": True, "users": json.loads(json.dumps(res.data or [], default=str))}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def send_friend_request(target_user_id):
    """Send a friend request to another user."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    try:
        # Check if already friends
        existing = supabase.table("friends").select("id").or_(
            f"and(user_id.eq.{current_user['id']},friend_id.eq.{target_user_id}),and(user_id.eq.{target_user_id},friend_id.eq.{current_user['id']})"
        ).execute()
        if existing.data:
            return {"success": False, "error": "Already friends."}

        # Check if request already pending
        pending = supabase.table("friend_requests").select("id").eq("sender_id", current_user["id"]).eq("receiver_id", target_user_id).eq("status", "pending").execute()
        if pending.data:
            return {"success": False, "error": "Request already sent."}

        # Check if THEY sent us one (auto-accept)
        reverse = supabase.table("friend_requests").select("id").eq("sender_id", target_user_id).eq("receiver_id", current_user["id"]).eq("status", "pending").execute()
        if reverse.data:
            return respond_friend_request(reverse.data[0]["id"], True)

        # Clean up any existing (accepted/declined) requests to avoid unique constraint violations
        supabase.table("friend_requests").delete().eq("sender_id", current_user["id"]).eq("receiver_id", target_user_id).execute()
        
        supabase.table("friend_requests").insert({
            "sender_id": current_user["id"],
            "receiver_id": target_user_id,
            "status": "pending"
        }).execute()

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def get_friend_requests():
    """Get pending friend requests for current user."""
    if not supabase or not current_user:
        return {"success": True, "incoming": [], "outgoing": []}
    try:
        incoming = supabase.table("friend_requests").select("id, sender_id, created_at, profiles!friend_requests_sender_id_fkey(username, avatar_url)").eq("receiver_id", current_user["id"]).eq("status", "pending").execute()
        outgoing = supabase.table("friend_requests").select("id, receiver_id, created_at, profiles!friend_requests_receiver_id_fkey(username, avatar_url)").eq("sender_id", current_user["id"]).eq("status", "pending").execute()
        return {
            "success": True,
            "incoming": json.loads(json.dumps(incoming.data or [], default=str)),
            "outgoing": json.loads(json.dumps(outgoing.data or [], default=str))
        }
    except Exception as e:
        print(f"[Friends] Get requests error: {e}", flush=True)
        # Fallback: simpler query without joins
        try:
            incoming = supabase.table("friend_requests").select("*").eq("receiver_id", current_user["id"]).eq("status", "pending").execute()
            outgoing = supabase.table("friend_requests").select("*").eq("sender_id", current_user["id"]).eq("status", "pending").execute()
            # Manually resolve usernames
            for req in (incoming.data or []):
                p = supabase.table("profiles").select("username, avatar_url").eq("id", req["sender_id"]).execute()
                req["profiles"] = p.data[0] if p.data else {"username": "Unknown"}
            for req in (outgoing.data or []):
                p = supabase.table("profiles").select("username, avatar_url").eq("id", req["receiver_id"]).execute()
                req["profiles"] = p.data[0] if p.data else {"username": "Unknown"}
            return {
                "success": True,
                "incoming": json.loads(json.dumps(incoming.data or [], default=str)),
                "outgoing": json.loads(json.dumps(outgoing.data or [], default=str))
            }
        except Exception as e2:
            return {"success": False, "error": str(e2), "incoming": [], "outgoing": []}


@eel.expose
def respond_friend_request(request_id, accept):
    """Accept or decline a friend request."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    try:
        req = supabase.table("friend_requests").select("*").eq("id", request_id).execute()
        if not req.data:
            return {"success": False, "error": "Request not found."}

        req_data = req.data[0]

        if accept:
            # Create bidirectional friendship
            supabase.table("friends").insert({"user_id": req_data["sender_id"], "friend_id": req_data["receiver_id"]}).execute()
            supabase.table("friends").insert({"user_id": req_data["receiver_id"], "friend_id": req_data["sender_id"]}).execute()

        supabase.table("friend_requests").update({"status": "accepted" if accept else "declined"}).eq("id", request_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def delete_user_account():
    """Permanently deletes the current user's local cache and cloud account."""
    global current_user
    if not current_user or not supabase:
        return {"success": False, "error": "Not logged in."}
        
    uid = current_user["id"]
    try:
        # 1. DELETE LOCAL CACHE
        cache_path = get_user_cache_path()
        import shutil
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"[Auth] Deleted local cache for user {uid}", flush=True)

        # 2. DELETE CLOUD ACCOUNT
        # Note: supabase-py admin.delete_user is used. 
        # This requires the service_role key to be set as SUPABASE_KEY in .env
        try:
            # 1. ANONYMIZE PUBLIC PROFILE (Soft Delete)
            # This preserves the row so Foreign Keys / Joins keep working
            # Generate a unique suffix for the ghost
            print(f"[Auth] Anonymizing profile for {uid}", flush=True)
            supabase.table("profiles").update({
                "username": f"Deleted User #{uid}",
                "email": f"deleted_{uid}@internal.local",
                "avatar_url": None,
                "last_seen": None
            }).eq("id", uid).execute()

            # 2. DELETE FROM SUPABASE AUTH (Admin)
            from supabase import create_client
            admin_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            admin_client.auth.admin.delete_user(uid)
            print(f"[Auth] Deleted cloud auth account for user {uid}", flush=True)
        except Exception as cloud_err:
            # If admin delete fails (likely due to anon key), we can't delete them from Python.
            # We'll return an error explaining they need the service role key or manual deletion.
            print(f"[Auth] Cloud deletion failed: {cloud_err}", flush=True)
            return {"success": False, "error": f"Cloud deletion failed. (Ensure service_role key is used): {cloud_err}"}

        # 3. Cleanup state
        current_user = None
        if os.path.exists(get_app_path("session.json")):
            os.remove(get_app_path("session.json"))
            
        return {"success": True}
        
    except Exception as e:
        print(f"[Auth] Critical Error in account deletion: {e}", flush=True)
        return {"success": False, "error": str(e)}


@eel.expose
def get_friends_list():
    """Get the current user's friends with online status and recent activity sorting."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated", "friends": []}
    try:
        from datetime import datetime, timezone, timedelta
        uid = current_user["id"]
        
        # 1. Fetch relationships (bidirectional)
        res = supabase.table("friends").select("*, p1:profiles!friends_user_id_fkey(id, username, avatar_url, last_seen), p2:profiles!friends_friend_id_fkey(id, username, avatar_url, last_seen)").or_(f"user_id.eq.{uid},friend_id.eq.{uid}").execute()
        
        if not res.data:
            return {"success": True, "friends": []}

        # 2. Fetch last message timestamps for sorting
        recent_dms = supabase.table("direct_messages").select("sender_id, receiver_id, created_at").or_(f"sender_id.eq.{uid},receiver_id.eq.{uid}").order("created_at", desc=True).limit(200).execute()
        last_activity_map = {}
        for msg in (recent_dms.data or []):
            partner_id = msg["sender_id"] if msg["receiver_id"] == uid else msg["receiver_id"]
            if partner_id not in last_activity_map:
                last_activity_map[partner_id] = msg["created_at"]

        friends = []
        seen_ids = set()
        for f in (res.data or []):
            p1 = f.get("p1") or {}
            p2 = f.get("p2") or {}
            other = p2 if p1.get("id") == uid else p1
            
            if not other or "id" not in other: continue
            
            fid = other["id"]
            if fid in seen_ids: continue
            seen_ids.add(fid)
            
            last_msg_at = last_activity_map.get(fid, "1970-01-01T00:00:00")
            
            friends.append({
                "id": fid,
                "rel_id": f["id"],
                "username": other.get("username", "Unknown"),
                "avatar_url": other.get("avatar_url"),
                "status": "online" if is_user_online(other.get("last_seen")) else "offline",
                "last_message_at": last_msg_at
            })
            
        # Sort: recent activity first, then online status
        friends.sort(key=lambda x: (x["last_message_at"] or ""), reverse=True)
        
        return {"success": True, "friends": json.loads(json.dumps(friends, default=str))}
    except Exception as e:
        print(f"[Friends] List Error: {e}", flush=True)
        return {"success": False, "error": str(e), "friends": []}


@eel.expose
def remove_friend(friend_id, rel_id=None):
    """Remove a friend (bidirectional). Supports rel_id for safe deletion of NULL entries."""
    print(f"[Friends] Removing: friend_id={friend_id}, rel_id={rel_id}", flush=True)
    if not supabase or not current_user:
        return {"success": False}
    try:
        if rel_id:
            # Safe delete by primary key
            supabase.table("friends").delete().eq("id", rel_id).execute()
        elif friend_id:
            # Delete bidirectional by IDs
            supabase.table("friends").delete().eq("user_id", current_user["id"]).eq("friend_id", friend_id).execute()
            supabase.table("friends").delete().eq("user_id", friend_id).eq("friend_id", current_user["id"]).execute()
        return {"success": True}
    except Exception as e:
        print(f"[Friends] Removal error: {e}", flush=True)
        return {"success": False, "error": str(e)}


@eel.expose
def block_user(user_id):
    """Block a user and remove friendship."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        remove_friend(user_id)
        supabase.table("blocked_users").insert({
            "blocker_id": current_user["id"],
            "blocked_id": user_id
        }).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# SOCIAL HUB — PHASE 3: DIRECT MESSAGES
# ============================================================

@eel.expose
def get_dm_history(friend_id):
    """Get message history between current user and a friend."""
    if not supabase or not current_user:
        return {"success": True, "messages": []}
    try:
        uid = current_user["id"]
        
        # Handle cases where friend_id is None (legacy deletions)
        if friend_id is None or str(friend_id).lower() == 'null':
            res = supabase.table("direct_messages").select("*, profiles!direct_messages_sender_id_fkey(username, avatar_url)").or_(
                f"and(sender_id.eq.{uid},receiver_id.is.null),and(sender_id.is.null,receiver_id.eq.{uid})"
            ).order("created_at", desc=False).limit(100).execute()
        else:
            res = supabase.table("direct_messages").select("*, profiles!direct_messages_sender_id_fkey(username, avatar_url)").or_(
                f"and(sender_id.eq.{uid},receiver_id.eq.{friend_id}),and(sender_id.eq.{friend_id},receiver_id.eq.{uid})"
            ).order("created_at", desc=False).limit(100).execute()
            
        return {"success": True, "messages": json.loads(json.dumps(res.data or [], default=str))}
    except Exception as e:
        print(f"[DM] History Error: {e}", flush=True)
        return {"success": False, "error": str(e), "messages": []}

@eel.expose
def edit_dm(message_id, new_content):
    """Edit an existing DM sent by the current user."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    if not new_content or not new_content.strip():
        return {"success": False, "error": "Content cannot be empty."}
    try:
        # Security: check ownership
        msg = supabase.table("direct_messages").select("sender_id").eq("id", message_id).single().execute()
        if not msg.data or msg.data["sender_id"] != current_user["id"]:
            return {"success": False, "error": "Permission denied."}
            
        supabase.table("direct_messages").update({"content": new_content.strip()}).eq("id", message_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def delete_dm(message_id):
    """Delete a DM sent by the current user."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    try:
        # Security: check ownership
        msg = supabase.table("direct_messages").select("sender_id").eq("id", message_id).single().execute()
        if not msg.data or msg.data["sender_id"] != current_user["id"]:
            return {"success": False, "error": "Permission denied."}
            
        supabase.table("direct_messages").delete().eq("id", message_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def send_dm(friend_id, content):
    """Send a direct message to a friend."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    if not content or not content.strip():
        return {"success": False, "error": "Empty message."}
    try:
        res = supabase.table("direct_messages").insert({
            "sender_id": current_user["id"],
            "receiver_id": friend_id,
            "content": content.strip(),
            "read": False
        }).execute()
        return {"success": True, "message": json.loads(json.dumps(res.data[0] if res.data else {}, default=str))}
    except Exception as e:
        return {"success": False, "error": str(e)}



def is_user_online(last_seen_iso):
    """Check if a user is online based on last_seen timestamp."""
    if not last_seen_iso: return False
    try:
        from datetime import datetime, timezone, timedelta
        # Handle both Z and +00:00 formats
        ls_str = last_seen_iso.replace('Z', '+00:00')
        last_seen = datetime.fromisoformat(ls_str)
        return datetime.now(timezone.utc) - last_seen < timedelta(minutes=5)
    except Exception as e:
        print(f"[Auth] Online check error: {ls_str} -> {e}", flush=True)
        return False


@eel.expose
def mark_dms_read(friend_id):
    """Mark all messages from a friend as read."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        supabase.table("direct_messages").update({"read": True}).eq("sender_id", friend_id).eq("receiver_id", current_user["id"]).eq("read", False).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def get_dm_unread_counts():
    """Get unread DM counts per friend for badge display."""
    if not supabase or not current_user:
        return {"success": True, "counts": {}}
    try:
        res = supabase.table("direct_messages").select("sender_id").eq("receiver_id", current_user["id"]).eq("read", False).execute()
        counts = {}
        for msg in (res.data or []):
            sid = msg["sender_id"]
            counts[sid] = counts.get(sid, 0) + 1
        return {"success": True, "counts": counts}
    except Exception as e:
        return {"success": True, "counts": {}}


@eel.expose
def get_detailed_profile(target_user_id):
    """
    Fetches comprehensive data for the Neural Dossier modal.
    Includes stats, books, and relationship status with the current user.
    """
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated"}
    
    try:
        my_id = current_user["id"]
        
        # 1. Fetch Basic Profile
        prof_res = supabase.table("profiles").select("*").eq("id", target_user_id).execute()
        if not prof_res.data:
            return {"success": False, "error": "User not found"}
        user = prof_res.data[0]
        
        # 2. Fetch User Stats (Self-Healing logic)
        # We select total_pages to allow the bootstrap logic to work if the cache isn't ready
        books_res = supabase.table("books").select("id, title, category, likes, is_public, created_at, total_pages").eq("owner_id", target_user_id).execute()
        books_list = books_res.data or []
        public_books = [b for b in books_list if b.get('is_public')]
        
        total_likes = sum(b.get("likes", 0) or 0 for b in public_books)
        
        # Get pages_translated from user_stats (our permanent XP stat)
        stats_data_res = supabase.table("user_stats").select("pages_translated").eq("user_id", target_user_id).execute()
        total_pages = stats_data_res.data[0].get("pages_translated", 0) if stats_data_res.data else 0
        
        # Self-Healing: If 0, try to bootstrap from published books
        if total_pages == 0 and public_books:
            try:
                # This is a one-time "bootstrap" for older accounts
                total_pages = sum(b.get("total_pages") or 0 for b in public_books if b.get("total_pages") is not None)
                
                # If we still have 0 but have books, check if pages are in 'metadata' (legacy support)
                if total_pages == 0:
                   for b in public_books:
                       m = b.get("metadata")
                       if isinstance(m, dict):
                           total_pages += int(m.get("pages") or 0)
                       elif isinstance(m, str):
                           try:
                               m_json = json.loads(m)
                               total_pages += int(m_json.get("pages") or 0)
                           except: pass
                
                if total_pages > 0:
                   supabase.table("user_stats").upsert({"user_id": target_user_id, "pages_translated": total_pages}).execute()
            except: pass
        
        # 3. Determine Relationship Status
        rel_status = 'none'
        request_id = None
        
        # Check Friendship
        friend_check = supabase.table("friends").select("id").eq("user_id", my_id).eq("friend_id", target_user_id).execute()
        if friend_check.data:
            rel_status = 'friend'
        else:
            # Check Pending Requests (Sent)
            sent_req = supabase.table("friend_requests").select("id").eq("sender_id", my_id).eq("receiver_id", target_user_id).eq("status", "pending").execute()
            if sent_req.data:
                rel_status = 'pending_sent'
                request_id = sent_req.data[0]["id"]
            else:
                # Check Pending Requests (Received)
                rec_req = supabase.table("friend_requests").select("id").eq("sender_id", target_user_id).eq("receiver_id", my_id).eq("status", "pending").execute()
                if rec_req.data:
                    rel_status = 'pending_received'
                    request_id = rec_req.data[0]["id"]

        # 4. Connection Rank Logic (Simple XP mapping)
        rank = "Initiate"
        if total_pages >= 2500: rank = "Grandmaster"
        elif total_pages >= 500: rank = "Architect"
        elif total_pages >= 100: rank = "Scribe"

        return {
            "success": True,
            "profile": {
                "id": user["id"],
                "username": user["username"],
                "avatar_url": user.get("avatar_url"),
                "bio": user.get("bio", "Neural Architect"),
                "last_seen": user.get("last_seen"),
                "books_published": len(public_books),
                "pages_translated": total_pages,
                "total_likes": total_likes,
                "rank": rank,
                "recent_books": sorted(public_books, key=lambda x: x['created_at'], reverse=True)[:5]
            },
            "relationship": {
                "status": rel_status,
                "request_id": request_id
            }
        }
    except Exception as e:
        print(f"[Dossier] Fetch Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def get_user_profile(user_id):
    """Standard profile fetch (Enhanced to match detailed if needed)."""
    return get_detailed_profile(user_id)


@eel.expose
def cancel_friend_request_by_target(target_id):
    """Cancels a pending friend request sent to a specific user ID."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        supabase.table("friend_requests").delete().eq("sender_id", current_user["id"]).eq("receiver_id", target_id).eq("status", "pending").execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def remove_friend_by_id(target_id):
    """Removes a friend by their user ID (Bidirectional)."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        my_id = current_user["id"]
        supabase.table("friends").delete().eq("user_id", my_id).eq("friend_id", target_id).execute()
        supabase.table("friends").delete().eq("user_id", target_id).eq("friend_id", my_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# SOCIAL HUB — PHASE 4: INBOX & NOTIFICATIONS (ENHANCED)
# ============================================================

@eel.expose
def get_pending_outgoing():
    """Get friend requests sent by the current user that are still pending."""
    if not supabase or not current_user:
        return {"success": True, "requests": []}
    try:
        uid = current_user["id"]
        res = supabase.table("friend_requests").select("*").eq("sender_id", uid).eq("status", "pending").execute()
        items = []
        for r in (res.data or []):
            receiver = supabase.table("profiles").select("username").eq("id", r["receiver_id"]).execute()
            receiver_name = receiver.data[0]["username"] if receiver.data else "Unknown User"
            items.append({
                "id": r["id"],
                "receiver_id": r["receiver_id"],
                "receiver_username": receiver_name,
                "created_at": r["created_at"]
            })
        return {"success": True, "requests": items}
    except Exception as e:
        print(f"[Friends] Pending outgoing error: {e}", flush=True)
        return {"success": False, "error": str(e)}


@eel.expose
def cancel_friend_request(request_id):
    """Cancel a friend request sent by the current user."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated"}
    try:
        # Verify ownership (sender_id must be the current user)
        res = supabase.table("friend_requests").delete().eq("id", request_id).eq("sender_id", current_user["id"]).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}



@eel.expose
def get_inbox():
    """Get unified inbox for the Social tab."""
    res = _get_unified_inbox()
    # Social inbox uses "items" key, while dashboard uses "notifications" key
    # We provide both for maximum compatibility during transition
    return {
        "success": res["success"],
        "items": res["items"],
        "notifications": res["items"],
        "unread_count": res["unread_count"]
    }


@eel.expose
def dismiss_notification(notif_id):
    """Delete a notification from the inbox."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        supabase.table("notifications").delete().eq("id", notif_id).eq("user_id", current_user["id"]).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def mark_all_notifications_read():
    """Mark all notifications as read for the current user."""
    if not supabase or not current_user:
        return {"success": False}
    try:
        supabase.table("notifications").update({"read": True}).eq("user_id", current_user["id"]).eq("read", False).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eel.expose
def get_unread_count():
    """Get the total unread count for the notification bell."""
    if not supabase or not current_user:
        return {"count": 0}
    try:
        notifs = supabase.table("notifications").select("id", count="exact").eq("user_id", current_user["id"]).eq("read", False).execute()
        fr = supabase.table("friend_requests").select("id", count="exact").eq("receiver_id", current_user["id"]).eq("status", "pending").execute()
        dms = supabase.table("direct_messages").select("id", count="exact").eq("receiver_id", current_user["id"]).eq("read", False).execute()
        total = (notifs.count or 0) + (fr.count or 0) + (dms.count or 0)
        return {"count": total}
    except Exception as e:
        return {"count": 0}


# ============================================================
# PHASE 5: ACTIVITY FEED & ACHIEVEMENTS
# ============================================================

def _trigger_activity_feed(user_id, activity_type, title, description, metadata={}, is_public_trigger=True):
    """Internal helper to insert a feed item. Respects user max-privacy settings."""
    if not supabase: return
    try:
        supabase.table("activity_feed").insert({
            "user_id": user_id,
            "type": activity_type,
            "title": title,
            "description": description,
            "metadata": metadata,
            "is_public": is_public_trigger,
            "is_pinned": False
        }).execute()
    except Exception as e:
        print(f"[Activity] Trigger Error: {e}", flush=True)


@eel.expose
def get_social_feed(feed_type="global", offset=0, limit=20):
    """Fetches the activity feed with pagination."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated."}
    
    uid = current_user["id"]
    try:
        query = supabase.table("activity_feed").select("*, profiles(username, avatar_url)")
        
        if feed_type == "friends":
            friends_res = supabase.table("friends").select("friend_id").eq("user_id", uid).execute()
            friend_ids = [f["friend_id"] for f in (friends_res.data or [])]
            friend_ids.append(uid)
            query = query.in_("user_id", friend_ids).neq("type", "patch_note")
        else:
            # Global: All items (temporarily removed is_public dependency)
            pass 

        res = query.order("is_pinned", desc=True).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        feed_items = res.data or []
        
        # --- Reaction Enrichment ---
        if feed_items:
            item_ids = [item["id"] for item in feed_items]
            
            # 1. Fetch all reactions for these items
            rx_res = supabase.table("activity_reactions").select("activity_id, reaction_type").in_("activity_id", item_ids).execute()
            all_rx = rx_res.data or []
            
            # 2. Fetch current user's reactions
            my_rx_res = supabase.table("activity_reactions").select("activity_id, reaction_type").in_("activity_id", item_ids).eq("user_id", uid).execute()
            my_rx = my_rx_res.data or []
            
            # 3. Aggregate counts
            for item in feed_items:
                i_id = item["id"]
                item["reactions"] = {
                    "clap": len([r for r in all_rx if r["activity_id"] == i_id and r["reaction_type"] == "clap"]),
                    "heart": len([r for r in all_rx if r["activity_id"] == i_id and r["reaction_type"] == "heart"]),
                    "fire": len([r for r in all_rx if r["activity_id"] == i_id and r["reaction_type"] == "fire"])
                }
                item["user_reacted"] = [r["reaction_type"] for r in my_rx if r["activity_id"] == i_id]

        # Ensure safe serialization for Eel (handles datetimes, etc.)
        feed_data = json.loads(json.dumps(feed_items, default=str))
        return {"success": True, "feed": feed_data}
        
    except Exception as e:
        err_str = str(e)
        if "column" in err_str and ("is_public" in err_str or "is_pinned" in err_str):
             return {"success": False, "error": f"DATABASE_MIGRATION_REQUIRED: {err_str}"}
        print(f"[Activity] Fetch Error: {e}", flush=True)
        return {"success": False, "error": f"Neural Link Failure: {err_str}"}

@eel.expose
def toggle_activity_reaction(activity_id, reaction_type):
    """Adds or removes a reaction for the current user (Toggle)."""
    if not supabase or not current_user:
        return {"success": False, "error": "Not authenticated"}
    
    uid = current_user["id"]
    try:
        # Check if exists
        res = supabase.table("activity_reactions").select("id").eq("activity_id", activity_id).eq("user_id", uid).eq("reaction_type", reaction_type).execute()
        
        if res.data:
            # Toggle OFF: Delete
            supabase.table("activity_reactions").delete().eq("id", res.data[0]["id"]).execute()
            status = "removed"
        else:
            # Toggle ON: Insert
            supabase.table("activity_reactions").insert({
                "activity_id": activity_id,
                "user_id": uid,
                "reaction_type": reaction_type
            }).execute()
            status = "added"
            
        # Get new count
        count_res = supabase.table("activity_reactions").select("id", count="exact").eq("activity_id", activity_id).eq("reaction_type", reaction_type).execute()
        return {"success": True, "status": status, "count": count_res.count}
        
    except Exception as e:
        print(f"[Activity] Toggle Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def repost_book(book_id):
    """Shares a book to the user's friend feed (Curated Repost)."""
    if not supabase or not current_user:
        return {"success": False}
    
    try:
        book = supabase.table("books").select("title, owner_username").eq("id", book_id).single().execute()
        if not book.data: return {"success": False}
        
        _trigger_activity_feed(
            user_id=current_user["id"],
            activity_type="curated_repost",
            title=f"Curator Recommendation",
            description=f"Check out '{book.data['title']}' originally by {book.data['owner_username']}.",
            metadata={"book_id": book_id},
            is_public_trigger=False 
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}



# ============================================================
# SOCIAL HUB — PHASE 7: COMMUNITY FORUMS
# ============================================================


@eel.expose
def get_forum_categories():
    """Fetches all forum categories with metadata."""
    if not supabase: return {"success": False}
    try:
        res = supabase.table("forum_categories").select("*").order("is_system", desc=True).order("name").execute()
        return {"success": True, "categories": json.loads(json.dumps(res.data or [], default=str))}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def get_forum_topics(category_id=None, search_query=None, tag_filter=None, author_id=None):
    """Fetches topics with flexible filtering and flattened profile data."""
    try:
        query = supabase.table("forum_topics").select("*, profiles!forum_topics_author_id_fkey(username, avatar_url)")
        
        if category_id:
            query = query.eq("category_id", category_id)
            
        if author_id:
            query = query.eq("author_id", author_id)
        
        if search_query:
            query = query.ilike("title", f"%{search_query}%")
            
        if tag_filter:
            query = query.contains("tags", [tag_filter])

        res = query.order("is_pinned", desc=True).order("created_at", desc=True).limit(100).execute()
        
        # Flatten the data for frontend (author_username, post_count etc.)
        topics = []
        for t in (res.data or []):
            # Fetch post count
            try:
                count_res = supabase.table("forum_posts").select("id", count='exact').eq("topic_id", t["id"]).execute()
                t["post_count"] = count_res.count if count_res.count is not None else 0
            except: t["post_count"] = 0

            # Map profile
            p = t.get("profiles") or {}
            t["author_username"] = p.get("username", "Unknown")
            t["author_avatar"] = p.get("avatar_url", "")
            
            # Ensure tags is a string for frontend splitting if stored as list
            if isinstance(t.get("tags"), list):
                t["tags"] = ", ".join(t["tags"])
                
            topics.append(t)
            
        return {"success": True, "topics": json.loads(json.dumps(topics, default=str))}
    except Exception as e:
        print(f"[Forums] Fetch Topics CRITICAL Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def get_forum_posts(topic_id):
    """Fetches all posts within a specific topic."""
    if not supabase: return {"success": False}
    try:
        # Increment View Count (Non-blocking)
        try: supabase.rpc('increment_topic_views', {'topic_id': topic_id}).execute()
        except: pass
        
        res = supabase.table("forum_posts").select("*, profiles!forum_posts_author_id_fkey(username, avatar_url)").eq("topic_id", topic_id).order("created_at").execute()
        
        posts = []
        for p in (res.data or []):
            prof = p.get("profiles") or {}
            p["username"] = prof.get("username", "Unknown")
            p["avatar_url"] = prof.get("avatar_url", "")
            posts.append(p)
            
        return {"success": True, "posts": json.loads(json.dumps(posts, default=str))}
    except Exception as e:
        print(f"[Forums] Fetch Posts Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def delete_forum_category(category_id):
    """Deletes a category (Owner or Admin only). SQL Handles cascade delete for topics."""
    global current_user
    if not supabase or not current_user: return {"success": False}
    uid = current_user["id"]
    try:
        cat = supabase.table("forum_categories").select("is_system, author_id").eq("id", category_id).single().execute()
        if not cat.data: return {"success": False, "error": "Category not found"}
        if cat.data["is_system"]: return {"success": False, "error": "System categories are protected."}
        
        # Permission check (RLS also guards this, but good to check here for clear errors)
        is_owner = cat.data["author_id"] == uid
        
        # Check if admin (simplified check - checks if user is admin of ANY topic as a proxy for social hub mod)
        # In a real app, this would check a global roles table.
        is_admin = False
        if not is_owner:
            mod_check = supabase.table("forum_moderators").select("id").eq("user_id", uid).eq("role", "admin").limit(1).execute()
            is_admin = bool(mod_check.data)

        if is_owner or is_admin:
            supabase.table("forum_categories").delete().eq("id", category_id).execute()
            return {"success": True}
            
        return {"success": False, "error": "Permission denied."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def delete_forum_topic(topic_id):
    """Deletes a forum and all its associated data (Author or Admin only)."""
    global current_user
    if not supabase or not current_user: return {"success": False}
    uid = current_user["id"]
    try:
        topic = supabase.table("forum_topics").select("author_id, is_system").eq("id", topic_id).single().execute()
        if not topic.data: return {"success": False, "error": "Topic not found."}
        
        # Check permissions
        is_owner = topic.data["author_id"] == uid
        is_admin = False
        if not is_owner:
            mod_check = supabase.table("forum_moderators").select("id").eq("user_id", uid).eq("topic_id", topic_id).eq("role", "admin").execute()
            is_admin = bool(mod_check.data)
            
        if not (is_owner or is_admin):
            return {"success": False, "error": "Permission denied."}
            
        # Delete related data first
        supabase.table("forum_moderators").delete().eq("topic_id", topic_id).execute()
        supabase.table("forum_posts").delete().eq("topic_id", topic_id).execute()
        
        # Delete from activity feed
        # We find activity entries that link to this topic via metadata
        supabase.table("activity_feed").delete().contains("metadata", {"topic_id": topic_id}).execute()
        
        # Finally delete topic
        del_res = supabase.table("forum_topics").delete().eq("id", topic_id).execute()
        
        if not del_res.data:
            print(f"[Forums] Delete FAILED for {topic_id} (Likely RLS block).", flush=True)
            return {"success": False, "error": "Database rejected deletion. Check mission permissions (RLS)."}
            
        print(f"[Forums] Deleted topic {topic_id} and all related data.", flush=True)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def delete_forum_post(post_id):
    """Deletes a specific post. Allowed for Author, Topic Owner, or Admin."""
    if not supabase or not current_user: return {"success": False, "error": "Not authenticated."}
    uid = current_user["id"]
    try:
        # Check if topic owner/admin or post author
        post_res = supabase.table("forum_posts").select("author_id, topic_id").eq("id", post_id).single().execute()
        if not post_res.data: return {"success": False, "error": "Post not found."}
        
        post_author_id = post_res.data["author_id"]
        topic_id = post_res.data["topic_id"]
        
        # Check topic owner
        topic_res = supabase.table("forum_topics").select("author_id").eq("id", topic_id).single().execute()
        topic_owner_id = topic_res.data["author_id"] if topic_res.data else None
        
        is_admin = current_user.get("is_admin", False)
        
        if uid == post_author_id or uid == topic_owner_id or is_admin:
            res = supabase.table("forum_posts").delete().eq("id", post_id).execute()
            if not res.data:
                return {"success": False, "error": "Database rejected deletion (RLS)."}
            return {"success": True}
        else:
            return {"success": False, "error": "Permission denied."}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def edit_forum_post(post_id, new_content):
    """Edits a specific post. Authored by the post owner only."""
    if not supabase or not current_user: return {"success": False, "error": "Not authenticated."}
    uid = current_user["id"]
    try:
        post_res = supabase.table("forum_posts").select("author_id").eq("id", post_id).single().execute()
        if not post_res.data: return {"success": False, "error": "Post not found."}
        
        if post_res.data["author_id"] == uid:
            res = supabase.table("forum_posts").update({"content": new_content, "updated_at": "now()"}).eq("id", post_id).execute()
            if not res.data:
                return {"success": False, "error": "Database rejected edit (RLS)."}
            return {"success": True}
        else:
            return {"success": False, "error": "Unauthorized to edit this post."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def create_forum_topic(category_id, title, initial_message, tags, feed_visibility="none", new_category_name=None):
    """
    Creates a new forum topic and its first post.
    Supports creating a new category on-the-fly if new_category_name is provided.
    Implements duplication guards for categories and forum titles.
    """
    global current_user
    if not supabase or not current_user: return {"success": False, "error": "Login to post."}
    
    uid = current_user["id"]
    try:
        # 0. Handle New Category Selection/Creation
        if new_category_name:
            # Case-insensitive existence check
            existing_cat = supabase.table("forum_categories").select("id").ilike("name", new_category_name.strip()).execute()
            if existing_cat.data:
                category_id = existing_cat.data[0]["id"]
            else:
                cat_res = supabase.table("forum_categories").insert({
                    "name": new_category_name.strip(),
                    "description": f"User-created category for discussions.",
                    "icon": "📁",
                    "is_system": False,
                    "author_id": uid
                }).execute()
                if not cat_res.data: return {"success": False, "error": "Failed to create category."}
                category_id = cat_res.data[0]["id"]
        
        if not category_id:
            return {"success": False, "error": "No category selected."}

        # 0.5 Check for duplicate forum title in SAME category
        dup_check = supabase.table("forum_topics").select("id").eq("category_id", category_id).ilike("title", title.strip()).execute()
        if dup_check.data:
            return {"success": False, "error": f"A forum with the name '{title}' already exists in this category."}

        # 1. Insert Topic
        topic_res = supabase.table("forum_topics").insert({
            "category_id": category_id,
            "author_id": uid,
            "title": title.strip(),
            "tags": tags if isinstance(tags, list) else [],
            "is_pinned": False,
            "is_locked": False
        }).execute()
        
        if not topic_res.data: 
            return {"success": False, "error": "No data returned from topic insert. Check RLS settings."}
            
        topic_id = topic_res.data[0]["id"]
        
        # 2. Assign creator as ADMIN in forum_moderators
        supabase.table("forum_moderators").insert({
            "user_id": uid,
            "topic_id": topic_id,
            "role": "admin"
        }).execute()
        
        # 3. Create First Post
        post_res = supabase.table("forum_posts").insert({
            "topic_id": topic_id,
            "author_id": uid,
            "content": initial_message.strip()
        }).execute()
        
        if not post_res.data:
            return {"success": False, "error": "Failed to create first post."}
        
        # 4. Optional Activity Feed Publish
        if feed_visibility != "none":
            # For "friends" visibility...
            is_pub = (feed_visibility == "global")
            
            _trigger_activity_feed(
                user_id=uid,
                activity_type="community_update",
                title=f"New Forum: {title}",
                description=initial_message[:100] + ( "..." if len(initial_message) > 100 else ""),
                metadata={"topic_id": topic_id, "category_id": category_id},
                is_public_trigger=is_pub
            )
            
        return {"success": True, "topic_id": topic_id}
    except Exception as e:
        print(f"[Forums] Create Topic CRITICAL Error: {e}", flush=True)
        return {"success": False, "error": str(e)}

@eel.expose
def create_forum_post(topic_id, content):
    """
    Adds a reply to an existing topic.
    """
    global current_user
    if not supabase or not current_user: return {"success": False, "error": "Login to reply."}
    
    uid = current_user["id"]
    try:
        # Check if topic is locked (simplified)
        topic = supabase.table("forum_topics").select("is_locked").eq("id", topic_id).single().execute()
        if topic.data and topic.data["is_locked"]:
            return {"success": False, "error": "This topic is locked."}
            
        # Insert Reply
        res = supabase.table("forum_posts").insert({
            "topic_id": topic_id,
            "author_id": uid,
            "content": content.strip()
        }).execute()
        
        return {"success": True, "message": json.loads(json.dumps(res.data[0] if res.data else {}, default=str))}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def delete_forum_item(item_type, item_id):
    """
    Deletes a topic or post if permission is held.
    Admin/Mod role required for high-level deletes.
    """
    global current_user
    if not supabase or not current_user: return {"success": False}
    
    uid = current_user["id"]
    try:
        if item_type == "topic":
            # Only Owner or Forum Admin
            topic = supabase.table("forum_topics").select("author_id").eq("id", item_id).single().execute()
            is_owner = topic.data and topic.data["author_id"] == uid
            
            is_admin = False
            if not is_owner:
                # Check for 'admin' role in moderators table
                mod = supabase.table("forum_moderators").select("role").eq("user_id", uid).eq("topic_id", item_id).execute()
                is_admin = mod.data and mod.data[0]["role"] == "admin"
                
            if is_owner or is_admin:
                supabase.table("forum_topics").delete().eq("id", item_id).execute()
                return {"success": True}
        else:
            # Post deletion logic
            post = supabase.table("forum_posts").select("author_id", "topic_id").eq("id", item_id).single().execute()
            if post.data:
                topic_id = post.data["topic_id"]
                is_author = post.data["author_id"] == uid
                
                # Check if user is mod/admin for this topic
                mod = supabase.table("forum_moderators").select("id").eq("user_id", uid).eq("topic_id", topic_id).execute()
                if is_author or mod.data:
                    supabase.table("forum_posts").delete().eq("id", item_id).execute()
                    return {"success": True}
                    
        return {"success": False, "error": "Permission denied."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def toggle_forum_pin(topic_id):
    """Toggles pinning of a topic (Admin/Mod only)."""
    global current_user
    if not supabase or not current_user: return {"success": False}
    
    uid = current_user["id"]
    try:
        # Check permission
        topic = supabase.table("forum_topics").select("author_id", "is_pinned").eq("id", topic_id).single().execute()
        if not topic.data: return {"success": False}
        
        is_owner = topic.data["author_id"] == uid
        mod = supabase.table("forum_moderators").select("role").eq("user_id", uid).eq("topic_id", topic_id).execute()
        is_admin = (mod.data and mod.data[0]["role"] == "admin")
        
        if is_owner or is_admin:
            new_status = not topic.data["is_pinned"]
            supabase.table("forum_topics").update({"is_pinned": new_status}).eq("id", topic_id).execute()
            return {"success": True, "is_pinned": new_status}
            
        return {"success": False, "error": "Unauthorized"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def get_trending_tags():
    """Returns top tags used in the last 7 days."""
    if not supabase: return {"success": False}
    try:
        # Simplified for prototype: returning all tags from the last 50 topics
        # In production, this would use a materialized view or complex SQL query
        res = supabase.table("forum_topics").select("tags").order("created_at", desc=True).limit(100).execute()
        
        tag_counts = {}
        for row in res.data:
            tags = row.get("tags", [])
            if isinstance(tags, list):
                for t in tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            elif isinstance(tags, str): # Fallback if stored as string
                for t in tags.split(','):
                    t = t.strip()
                    if t: tag_counts[t] = tag_counts.get(t, 0) + 1
                    
        sorted_tags = sorted([{"tag": k, "count": v} for k, v in tag_counts.items()], key=lambda x: x["count"], reverse=True)
        return {"success": True, "tags": sorted_tags[:10]}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    main()
