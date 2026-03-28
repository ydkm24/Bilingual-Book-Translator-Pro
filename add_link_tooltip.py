import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"
html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

# --- CSS ADDITION ---
with open(css_path, "a", encoding="utf-8") as f:
    f.write("""
/* Link Tooltip */
.link-tooltip {
    position: fixed;
    z-index: 10000;
    background: #1a1a1a;
    border: 1px solid var(--accent-green);
    border-radius: 6px;
    padding: 4px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    font-size: 12px;
    color: #fff;
    pointer-events: auto;
}
.link-tooltip a {
    color: var(--accent-green);
    text-decoration: underline;
    max-width: 150px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.link-tooltip button {
    background: none;
    border: 1px solid rgba(255,255,255,0.2);
    color: #acc;
    padding: 2px 6px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
}
.link-tooltip button:hover {
    background: rgba(255,255,255,0.1);
    color: #fff;
}
.link-tooltip button.btn-remove:hover {
    border-color: #ef4444;
    color: #ef4444;
}
""")

# --- HTML/JS ADDITION ---
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Add the tooltip element
html = html.replace('</body>', """
    <!-- Link Settings Tooltip -->
    <div id="link-tooltip" class="link-tooltip" style="display:none;">
        <a id="link-tooltip-url" href="#" target="_blank">URL</a>
        <div style="width:1px; height:12px; background:rgba(255,255,255,0.2);"></div>
        <button class="btn-remove" onclick="removeActiveLink()">Remove</button>
    </div>
</body>""")

# Add the JS logic
js_logic = """
        // --- LINK TOOLTIP LOGIC ---
        let _activeLinkNode = null;
        function setupLinkTooltip() {
            document.addEventListener('mouseover', (e) => {
                const link = e.target.closest('a');
                if (link && link.closest('.docs-pane-right')) {
                    showLinkTooltip(link);
                }
            });
            
            // Hide if mouse leaves both link and tooltip
            document.addEventListener('mousemove', (e) => {
                const tooltip = document.getElementById('link-tooltip');
                if (!tooltip || tooltip.style.display === 'none') return;
                
                const isOverLink = e.target.closest('a') === _activeLinkNode;
                const isOverTooltip = e.target.closest('#link-tooltip');
                
                if (!isOverLink && !isOverTooltip) {
                    hideLinkTooltip();
                }
            });
        }

        function showLinkTooltip(link) {
            const tooltip = document.getElementById('link-tooltip');
            const urlEl = document.getElementById('link-tooltip-url');
            if(!tooltip || !urlEl) return;
            
            _activeLinkNode = link;
            urlEl.href = link.href;
            urlEl.textContent = link.href;
            
            tooltip.style.display = 'flex';
            const rect = link.getBoundingClientRect();
            tooltip.style.top = (rect.top - 35) + 'px';
            tooltip.style.left = rect.left + 'px';
        }

        function hideLinkTooltip() {
            const tooltip = document.getElementById('link-tooltip');
            if(tooltip) tooltip.style.display = 'none';
            _activeLinkNode = null;
        }

        function removeActiveLink() {
            if (_activeLinkNode) {
                // Select the link text
                const sel = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(_activeLinkNode);
                sel.removeAllRanges();
                sel.addRange(range);
                
                document.execCommand('unlink', false, null);
                hideLinkTooltip();
                markNeuralDocsUnsaved();
            }
        }
"""

# Find a good place to inject the JS logic (e.g., after initRulerForContainer)
inject_marker = "function initRulerForContainer"
if inject_marker in html:
    html = html.replace(inject_marker, js_logic + "\n        " + inject_marker)

# Call setupLinkTooltip in DOMContentLoaded
html = html.replace('window.addEventListener(\'DOMContentLoaded\', () => {', 'window.addEventListener(\'DOMContentLoaded\', () => {\n            setupLinkTooltip();')

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Link tooltip helper added.")
