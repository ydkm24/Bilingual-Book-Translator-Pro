import os

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add listeners to the rightPane creation
old_listeners = """                rightPane.addEventListener('input', () => {
                    markNeuralDocsUnsaved();
                });"""

new_listeners = """                rightPane.addEventListener('input', () => {
                    markNeuralDocsUnsaved();
                });
                rightPane.addEventListener('mouseup', saveSelection);
                rightPane.addEventListener('keyup', saveSelection);"""

html = html.replace(old_listeners, new_listeners)

# 2. Update saveSelection and restoreAndApplyColor
old_func = """        let _savedRange = null;
        function saveSelection() {
            const sel = window.getSelection();
            if (sel.rangeCount > 0) {
                _savedRange = sel.getRangeAt(0).cloneRange();
            }
        }
        function restoreAndApplyColor(command, value) {
            if (_savedRange) {
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(_savedRange);
            }
            document.execCommand(command, false, value);
            _savedRange = null;
        }"""

new_func = """        let _savedRange = null;
        function saveSelection() {
            const sel = window.getSelection();
            if (sel.rangeCount > 0) {
                const range = sel.getRangeAt(0);
                // Only save if the selection is inside a docs-pane-right
                let node = range.commonAncestorContainer;
                while(node && node !== document.body) {
                    if(node.classList && node.classList.contains('docs-pane-right')) {
                        _savedRange = range.cloneRange();
                        break;
                    }
                    node = node.parentNode;
                }
            }
        }
        function restoreAndApplyColor(command, value) {
            if (_savedRange) {
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(_savedRange);
            }
            document.execCommand(command, false, value);
            // Don't null it immediately, in case they change color again without re-selecting
        }"""

html = html.replace(old_func, new_func)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Selection persistence logic improved.")
