import os

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update the JS promptInsertImage function and add handleImageUpload
old_func = """        function promptInsertImage() {
            const url = prompt("Enter image URL:");
            if (url) {
                document.execCommand('insertImage', false, url);
                // Try resizing the newly added image immediately
                setTimeout(() => {
                    const imgs = document.querySelectorAll('.docs-text-content img');
                    imgs.forEach(i => {
                        if(i.src === url) {
                            i.style.maxWidth = '100%';
                            i.style.height = 'auto';
                            i.style.borderRadius = '4px';
                        }
                    });
                }, 100);
            }
        }"""

new_func = """        function promptInsertImage() {
            const fileInput = document.getElementById('image-upload-input');
            if (fileInput) fileInput.click();
        }

        function handleImageUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                const dataUrl = event.target.result;
                document.execCommand('insertImage', false, dataUrl);
                
                // Add some premium formatting to the newly inserted image
                setTimeout(() => {
                    const imgs = document.querySelectorAll('.docs-pane-right img');
                    imgs.forEach(i => {
                        if(i.src === dataUrl) {
                            i.style.maxWidth = '100%';
                            i.style.height = 'auto';
                            i.style.borderRadius = '8px';
                            i.style.margin = '10px 0';
                            i.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                            i.style.display = 'block';
                        }
                    });
                    markNeuralDocsUnsaved();
                }, 50);
            };
            reader.readAsDataURL(file);
            e.target.value = ''; // Reset for same-file re-upload
        }"""

html = html.replace(old_func, new_func)

# 2. Add the hidden input to the HTML body
if 'id="image-upload-input"' not in html:
    html = html.replace('</body>', '<input type="file" id="image-upload-input" accept="image/*" style="display:none" onchange="handleImageUpload(event)">\n</body>')

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: Image insertion changed to local file upload.")
