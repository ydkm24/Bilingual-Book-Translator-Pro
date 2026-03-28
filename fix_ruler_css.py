import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"

with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

# 1. Update container
old_container = """.docs-pane-right-container {
    flex: 1 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: #ffffff !important;
    overflow: hidden !important;
}"""
new_container = """.docs-pane-right-container {
    flex: 1 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: #ffffff !important;
    overflow: hidden !important;
    position: relative !important;
}"""
css = css.replace(old_container, new_container)

# 2. Update ruler visuals
old_ruler = """.docs-ruler {
    height: 24px;
    background: #111;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: relative;
    user-select: none;
    display: block; /* We'll show it when the workspace opens */
}
.ruler-ticks {
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(to right, #444 1px, transparent 1px),
        linear-gradient(to right, #333 1px, transparent 1px);
    background-size: 100px 100%, 10px 100%;
    background-position: 0 100%, 0 100%;
    background-repeat: repeat-x;
    opacity: 0.5;
}
.ruler-ticks::before {
    content: "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16";
    word-spacing: 84.5px; /* Roughly 100px chunks */
    position: absolute;
    top: 2px;
    left: 102px;
    font-size: 10px;
    font-family: monospace;
    color: #666;
    letter-spacing: 1px;
    white-space: nowrap;
}"""

new_ruler = """.docs-ruler {
    height: 24px;
    background: #f8f9fa;
    border-bottom: 1px solid #e2e8f0;
    position: absolute;
    top: 0; left: 0; width: 100%;
    z-index: 10;
    user-select: none;
    display: block;
}
.ruler-ticks {
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(to right, #cbd5e1 1px, transparent 1px),
        linear-gradient(to right, #e2e8f0 1px, transparent 1px);
    background-size: 100px 100%, 10px 100%;
    background-position: 0 100%, 0 100%;
    background-repeat: repeat-x;
    opacity: 1;
}
.ruler-ticks::before {
    content: "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16";
    word-spacing: 84.5px;
    position: absolute;
    top: 2px;
    left: 102px;
    font-size: 10px;
    font-family: monospace;
    color: #94a3b8;
    letter-spacing: 1px;
    white-space: nowrap;
}"""
css = css.replace(old_ruler, new_ruler)

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css)

print("SUCCESS: Ruler CSS updated to overlay correctly and be white-themed.")
