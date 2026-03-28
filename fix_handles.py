import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"
md_path = r"c:\Users\TheGoyFather\.gemini\antigravity\brain\afa70f60-ec61-44d8-b1e9-ab1546b8acc5\walkthrough.md"

with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

# Replace variables with standard hex #3b82f6
css = css.replace('var(--accent-blue)', '#3b82f6')

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css)

with open(md_path, "r", encoding="utf-8") as f:
    md = f.read()

md = md.replace('file:///C:/Users/TheGoyFather/.gemini/antigravity/brain/afa70f60-ec61-44d8-b1e9-ab1546b8acc5/media__1774625513155.png', r'C:\Users\TheGoyFather\.gemini\antigravity\brain\afa70f60-ec61-44d8-b1e9-ab1546b8acc5\media__1774625513155.png')

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

print("SUCCESS: Handles patched with hardcoded blue color, markdown image path fixed.")
