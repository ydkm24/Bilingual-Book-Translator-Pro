import os, json, glob

cache_dir = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\.translator_cache\ydkm24"
if not os.path.exists(cache_dir):
    print("USER CACHE NOT FOUND")
else:
    projects = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and d != ".export_drafts"]
    if not projects:
        print("NO PROJECTS FOUND")
    else:
        book_id = projects[0]
        print(f"CHECKING PROJECT: {book_id}")
        proj_dir = os.path.join(cache_dir, book_id)
        pages = glob.glob(os.path.join(proj_dir, "page_*.json"))
        if not pages:
            print("NO PAGES FOUND")
        else:
            with open(pages[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                print(json.dumps(data, indent=2))
