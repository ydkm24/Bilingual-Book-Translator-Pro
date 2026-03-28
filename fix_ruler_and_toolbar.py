import os
import re

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"
css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. REMOVE the global docs-ruler from HTML
html = re.sub(r'<!-- Document Ruler -->\s*<div class="docs-ruler" id="docs-ruler">.*?</div>\s*</div>\s*<div class="docs-canvas" id="neural-docs-canvas">', '<div class="docs-canvas" id="neural-docs-canvas">', html, flags=re.DOTALL)
html = re.sub(r'<!-- Document Ruler -->\s*<div class="docs-ruler" id="docs-ruler" style="display:none;">.*?</div>\s*</div>\s*<div class="docs-canvas" id="neural-docs-canvas">', '<div class="docs-canvas" id="neural-docs-canvas">', html, flags=re.DOTALL)
html = re.sub(r'<!-- Document Ruler -->\s*<div class="docs-ruler" id="docs-ruler">.*?</div>(?:<!-- \n)*\s*<div class="docs-canvas" id="neural-docs-canvas">', '<div class="docs-canvas" id="neural-docs-canvas">', html, flags=re.DOTALL)

# Let's just do a manual string replace for docs-ruler if regex fails
if 'id="docs-ruler"' in html:
    start_idx = html.find('<!-- Document Ruler -->')
    end_idx = html.find('<div class="docs-canvas" id="neural-docs-canvas">')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        html = html[:start_idx] + html[end_idx:]


# 2. Add Ruler Template to HTML body
template_html = """
    <!-- Ruler Template -->
    <template id="ruler-template">
        <div class="docs-ruler" style="width:100%;">
            <div class="ruler-ticks"></div>
            <div class="ruler-handle indent-first" title="First Line Indent"></div>
            <div class="ruler-handle indent-left" title="Left Indent"></div>
            <div class="ruler-handle indent-right" title="Right Margin"></div>
        </div>
    </template>
"""
if 'id="ruler-template"' not in html:
    html = html.replace('</body>', template_html + '\n</body>')


# 3. Modify renderNeuralDocsPages to enclose right-pane
# Find the exact lines in renderNeuralDocsPages where right pane is built
old_rp = """                // RIGHT PANE: Editable Text
                const rightPane = document.createElement('div');
                rightPane.className = 'docs-pane-right';
                rightPane.contentEditable = "true";
                rightPane.spellcheck = false;
                rightPane.setAttribute('data-page-index', p.page_index);
                
                const text = p.english || p.translated || '';
                rightPane.innerHTML = text.replace(/\\n/g, '<br>');
                
                rightPane.addEventListener('input', () => {
                    markNeuralDocsUnsaved();
                });
                
                spread.appendChild(leftPane);
                spread.appendChild(splitter);
                spread.appendChild(rightPane);"""

new_rp = """                // RIGHT PANE: Container
                const rightContainer = document.createElement('div');
                rightContainer.className = 'docs-pane-right-container';
                
                // Add Ruler from Template
                const rulerTpl = document.getElementById('ruler-template');
                if(rulerTpl) {
                    const rulerNode = rulerTpl.content.cloneNode(true);
                    rightContainer.appendChild(rulerNode);
                }

                // RIGHT PANE: Editable Text
                const rightPane = document.createElement('div');
                rightPane.className = 'docs-pane-right docs-text-content';
                rightPane.contentEditable = "true";
                rightPane.spellcheck = false;
                rightPane.setAttribute('data-page-index', p.page_index);
                
                const text = p.english || p.translated || '';
                rightPane.innerHTML = text.replace(/\\n/g, '<br>');
                
                rightPane.addEventListener('input', () => {
                    markNeuralDocsUnsaved();
                });
                
                rightContainer.appendChild(rightPane);
                
                spread.appendChild(leftPane);
                spread.appendChild(splitter);
                spread.appendChild(rightContainer);
                
                // Initialize the ruler JS for this specific container
                setTimeout(() => {
                    initRulerForContainer(rightContainer, rightPane);
                }, 100);"""

if old_rp in html:
    html = html.replace(old_rp, new_rp)
else:
    print("WARNING: Could not find exact old_rp replacement block.")


# 4. Replace initRuler with initRulerForContainer in JS
js_old = """        // --- DOCUMENT RULER LOGIC ---
        function initRuler() {
            const ruler = document.getElementById('docs-ruler');
            const hFirst = document.getElementById('indent-first');
            const hLeft = document.getElementById('indent-left');
            const hRight = document.getElementById('indent-right');
            if(!ruler || !hFirst || !hLeft || !hRight) return;"""

js_new = """        // --- DOCUMENT RULER LOGIC ---
        function initRulerForContainer(container, textPane) {
            const ruler = container.querySelector('.docs-ruler');
            const hFirst = container.querySelector('.indent-first');
            const hLeft = container.querySelector('.indent-left');
            const hRight = container.querySelector('.indent-right');
            if(!ruler || !hFirst || !hLeft || !hRight) return;"""

if js_old in html:
    html = html.replace(js_old, js_new)
    # also remove window.addEventListener('DOMContentLoaded', () => initRuler());
    html = re.sub(r'window\.addEventListener\(\'DOMContentLoaded\',\s*\(\)\s*=>\s*\{\s*initRuler\(\);\s*\}\);', '', html)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

# --- CSS CHANGES ---
with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

if '.docs-pane-right-container' not in css:
    css_add = """
.docs-pane-right-container {
    flex: 1 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: #ffffff !important;
    overflow: hidden !important;
}
"""
    css += css_add

# modify docs-pane-right to not have flex:1 height 100%, and to remove padding-top so ruler sits flush
if '.docs-pane-right {' in css:
    css = css.replace('.docs-pane-right {\n    flex: 1 !important;\n    height: 100% !important;', '.docs-pane-right {\n    flex: 1 !important;\n    height: calc(100% - 24px) !important;')

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css)

print("SUCCESS: Fixed Ruler Placement!")
