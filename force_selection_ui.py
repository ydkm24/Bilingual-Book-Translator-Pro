import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"
html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(css_path, "a", encoding="utf-8") as f:
    f.write("""
/* Keep selection visible even when focus is lost (like opening a color picker) */
.docs-pane-right::selection {
    background: rgba(74, 222, 128, 0.4) !important;
    color: #000 !important;
}
.docs-pane-right:not(:focus)::selection {
    background: rgba(74, 222, 128, 0.4) !important;
    color: #000 !important;
}
""")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Update color pickers to use onclick as well just in case
html = html.replace('onmousedown="saveSelection()" onchange="restoreAndApplyColor', 'onmousedown="saveSelection()" onclick="saveSelection()" onchange="restoreAndApplyColor')

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Forced selection CSS and added click-backup for color pickers.")
