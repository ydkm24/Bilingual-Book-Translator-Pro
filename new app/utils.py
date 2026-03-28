"""
utils.py — Constants, helpers, and environment configuration for Bilingual Book Translator Pro.
Extracted from main.py during The Great Refactor.
"""
import os
import sys
from dotenv import load_dotenv

# --- SPRINT 37: GLOBAL CONFIG & PORTABILITY ---
APP_VERSION = "1.5.0"

def humanize_error(e):
    """Translates technical networking/API errors into plain English for the user."""
    err_str = str(e)
    # DNS / Connection Errors
    if "getaddrinfo failed" in err_str or "gaierror" in err_str:
        return "No Internet Connection (DNS lookup failed). Your Wi-Fi may be down or your Supabase project might be PAUSED."
    if "Connection refused" in err_str:
        return "Server unreachable. The service may be down, or your firewall is blocking the app."
    # API / Rate Limit Errors
    if "429" in err_str or "Too Many Requests" in err_str:
        return "Too many requests (Rate Limited). Please wait a minute and try again."
    if "Timeout" in err_str or "timed out" in err_str:
        return "The request timed out. Check your internet speed or VPN settings."
    # API Specific Failures
    if "OpenAI API Error" in err_str:
        return f"OpenAI Error: Ensure your API Key is valid and has credits.\n\nDetail: {err_str}"
    if "DeepL API Error" in err_str:
        return f"DeepL Error: Check your API Key and plan limit.\n\nDetail: {err_str}"
    
    return err_str

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_app_path(sub_path=""):
    """Get path to the user's AppData/Local folder for the app."""
    path = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), "PDF_Translator")
    
    # Always ensure the BASE directory exists
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    # Return joined path but DO NOT create it (might be a file)
    if sub_path:
        return os.path.join(path, sub_path)
    return path

# Load Supabase credentials
if getattr(sys, 'frozen', False):
    # If built with PyInstaller, use the bundled .env
    env_path = os.path.join(sys._MEIPASS, ".env")
    load_dotenv(env_path)
else:
    load_dotenv()

# True Light / Midnight Pro Palette Tuples (Light, Dark)
PAPER_BG = ("#FDF5E6", "#121212") # Beige / Deep matte black
TEXT_COLOR = ("#111111", "#E0E0E0") # Primary text
TEXT_DIM = ("#555555", "#A0A0A0") # Secondary text
HEADER_BG = ("#F5E6CC", "#1E1E1E") # Elevated frame background
ACCENT_COLOR = ("#0056b3", "#00D2FF") # Electric Blue
ACCENT_HOVER = ("#004494", "#00B4DB")
PAPER_SHEET_BG = ("#FFFFFF", "#242424") # Surface color for pages
BORDER_COLOR = ("#D4C4A8", "#333333") 
CARD_BG = ("#E8D5B5", "#2D2D2D")
FRAME_BG = ("#F5E6CC", "#1E1E1E")
BTN_PRIMARY = ("#6200EE", "#BB86FC")
BTN_PRIMARY_HOVER = ("#3700B3", "#9965f4")
BTN_SECONDARY = ("#E0E0E0", "#333333")
BTN_SECONDARY_HOVER = ("#d5c09e", "#444444")
INPUT_BG = ("#FFFFFF", "#2D2D2D")
BTN_DANGER = ("#E53935", "#C62828")
BTN_DANGER_HOVER = ("#D32F2F", "#B71C1C")
BTN_SUCCESS = ("#4CAF50", "#27AE60")
BTN_SUCCESS_HOVER = ("#45A049", "#2ECC71")
BTN_WARNING = ("#FFB74D", "#FF9800")
BTN_WARNING_HOVER = ("#FFA726", "#F57C00")
