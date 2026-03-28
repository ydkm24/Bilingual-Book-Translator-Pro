import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"
html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

# 1. Add table CSS
with open(css_path, "a", encoding="utf-8") as f:
    f.write("""
/* Editor Table Styling */
.docs-pane-right table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 15px 0 !important;
    border: 1px solid #dee2e6 !important;
    table-layout: fixed !important;
}
.docs-pane-right th, .docs-pane-right td {
    border: 1px solid #dee2e6 !important;
    padding: 12px !important;
    min-height: 40px !important;
    vertical-align: top !important;
    word-break: break-word !important;
}
""")

# 2. Update the JS insertBasicTable function
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

old_func = """        function insertBasicTable() {
            const tableHTML = `
                <table style="width:100%; border-collapse: collapse; border: 1px solid rgba(255,255,255,0.2); margin: 10px 0;">
                    <tbody>
                        <tr><td style="border: 1px solid rgba(255,255,255,0.2); padding: 8px;">Cell 1</td><td style="border: 1px solid rgba(255,255,255,0.2); padding: 8px;">Cell 2</td></tr>
                        <tr><td style="border: 1px solid rgba(255,255,255,0.2); padding: 8px;">Cell 3</td><td style="border: 1px solid rgba(255,255,255,0.2); padding: 8px;">Cell 4</td></tr>
                    </tbody>
                </table><p><br></p>`;
            document.execCommand('insertHTML', false, tableHTML);
        }"""

new_func = """        function insertBasicTable() {
            // Ensure selection is restored so it inserts at the cursor
            if (_savedRange) {
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(_savedRange);
            }
            
            const tableHTML = `
                <table style="width:100%; border-collapse: collapse; border: 1px solid #dee2e6; margin: 10px 0;">
                    <tbody>
                        <tr><td style="border: 1px solid #dee2e6; padding: 10px;"></td><td style="border: 1px solid #dee2e6; padding: 10px;"></td></tr>
                        <tr><td style="border: 1px solid #dee2e6; padding: 10px;"></td><td style="border: 1px solid #dee2e6; padding: 10px;"></td></tr>
                    </tbody>
                </table><p><br></p>`;
            document.execCommand('insertHTML', false, tableHTML);
            markNeuralDocsUnsaved();
        }"""

html = html.replace(old_func, new_func)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Table insertion fixed with visible borders and selection restoration.")
