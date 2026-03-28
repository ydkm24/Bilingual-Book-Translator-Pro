import os

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Re-insert Undo/Redo at the beginning of the lower toolbar
toolbar_start = '<div class="docs-lower-toolbar" id="docs-lower-toolbar">'
if toolbar_start in html:
    undo_redo = """
                         <!-- Action Group: Undo/Redo -->
                         <button class="tb" onclick="document.execCommand('undo',false,null)" title="Undo (Ctrl+Z)">
                             <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 14L4 9l5-5"/><path d="M20 20v-7a4 4 0 0 0-4-4H4"/></svg>
                         </button>
                         <button class="tb" onclick="document.execCommand('redo',false,null)" title="Redo (Ctrl+Y)">
                             <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 14l5-5-5-5"/><path d="M4 20v-7a4 4 0 0 1 4-4h12"/></svg>
                         </button>
                         <div class="tb-sep"></div>
"""
    html = html.replace(toolbar_start, toolbar_start + undo_redo)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Restored Undo/Redo buttons.")
