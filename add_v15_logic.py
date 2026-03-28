import os

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# --- 1. ADD OVERLAYS TO BODY ---
body_overlays = """
    <!-- Object Manipulation Overlays -->
    <div id="obj-resizer-overlay" class="obj-resizer-overlay">
        <div class="resizer-handle nw" data-handle="nw"></div>
        <div class="resizer-handle ne" data-handle="ne"></div>
        <div class="resizer-handle sw" data-handle="sw"></div>
        <div class="resizer-handle se" data-handle="se"></div>
    </div>
    
    <div id="obj-control-bubble" class="obj-control-bubble" style="display:none;"></div>
</body>"""
html = html.replace('</body>', body_overlays)

# --- 2. ADD JS LOGIC ---
obj_logic = """
        // --- V15: ADVANCED OBJECT MANIPULATION (IMAGES & TABLES) ---
        let _manipulatingObj = null;
        let _resizingImg = null;
        let _resizeStartX, _resizeStartY, _resizeStartW, _resizeStartH;

        function initObjectManipulation() {
            document.addEventListener('mousedown', (e) => {
                // If clicking outside, hide overlays (unless clicking on handles/bubble)
                if (!e.target.closest('img') && !e.target.closest('table') && 
                    !e.target.closest('#obj-resizer-overlay') && !e.target.closest('#obj-control-bubble')) {
                    hideObjectOverlays();
                }
            });

            document.addEventListener('click', (e) => {
                const img = e.target.closest('.docs-pane-right img');
                const table = e.target.closest('.docs-pane-right table');
                
                if (img) {
                    _manipulatingObj = img;
                    showImageControls(img);
                } else if (table) {
                    _manipulatingObj = table;
                    const cell = e.target.closest('td, th');
                    showTableControls(table, cell);
                }
            });

            // Handle Resizing
            const overlay = document.getElementById('obj-resizer-overlay');
            overlay.addEventListener('mousedown', (e) => {
                const handle = e.target.dataset.handle;
                if (!handle || !_manipulatingObj) return;
                
                e.preventDefault();
                _resizingImg = _manipulatingObj;
                _resizeStartX = e.clientX;
                _resizeStartY = e.clientY;
                _resizeStartW = _resizingImg.offsetWidth;
                _resizeStartH = _resizingImg.offsetHeight;
                
                const onMouseMove = (me) => {
                    const dx = me.clientX - _resizeStartX;
                    const dy = me.clientY - _resizeStartY;
                    
                    let nw, nh;
                    if (handle === 'se') { nw = _resizeStartW + dx; nh = _resizeStartH + dy; }
                    else if (handle === 'sw') { nw = _resizeStartW - dx; nh = _resizeStartH + dy; }
                    else if (handle === 'ne') { nw = _resizeStartW + dx; nh = _resizeStartH - dy; }
                    else if (handle === 'nw') { nw = _resizeStartW - dx; nh = _resizeStartH - dy; }
                    
                    if (nw > 20) _resizingImg.style.width = nw + 'px';
                    if (nh > 20) _resizingImg.style.height = nh + 'px';
                    
                    updateOverlayPos(_resizingImg);
                };
                
                const onMouseUp = () => {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    _resizingImg = null;
                    markNeuralDocsUnsaved();
                };
                
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });
        }

        function showImageControls(img) {
            hideObjectOverlays();
            updateOverlayPos(img);
            const overlay = document.getElementById('obj-resizer-overlay');
            overlay.style.display = 'block';
            
            const bubble = document.getElementById('obj-control-bubble');
            bubble.innerHTML = `
                <button class="obj-btn" onclick="alignObj('left')">Left</button>
                <button class="obj-btn" onclick="alignObj('center')">Center</button>
                <button class="obj-btn" onclick="alignObj('right')">Right</button>
                <div style="width:1px; background:rgba(255,255,255,0.2); margin:0 4px;"></div>
                <button class="obj-btn danger" onclick="deleteObj()">Delete</button>
            `;
            const rect = img.getBoundingClientRect();
            bubble.style.display = 'flex';
            bubble.style.top = (window.scrollY + rect.top - 45) + 'px';
            bubble.style.left = (window.scrollX + rect.left) + 'px';
        }

        function showTableControls(table, cell) {
            hideObjectOverlays();
            const bubble = document.getElementById('obj-control-bubble');
            bubble.innerHTML = `
                <button class="obj-btn" onclick="tableAction('addRow')">+ Row</button>
                <button class="obj-btn" onclick="tableAction('addCol')">+ Col</button>
                <div style="width:1px; background:rgba(255,255,255,0.2); margin:0 4px;"></div>
                <button class="obj-btn danger" onclick="deleteObj()">Delete Table</button>
            `;
            const rect = table.getBoundingClientRect();
            bubble.style.display = 'flex';
            bubble.style.top = (window.scrollY + rect.top - 45) + 'px';
            bubble.style.left = (window.scrollX + rect.left) + 'px';
            _activeCell = cell || table.querySelector('td');
        }

        let _activeCell = null;
        function tableAction(action) {
            if (!_manipulatingObj || !_activeCell) return;
            const table = _manipulatingObj;
            if (action === 'addRow') {
                const newRow = table.insertRow(_activeCell.parentElement.rowIndex + 1);
                for (let i = 0; i < _activeCell.parentElement.cells.length; i++) {
                    const c = newRow.insertCell();
                    c.style.cssText = _activeCell.style.cssText;
                    c.innerHTML = "";
                }
            } else if (action === 'addCol') {
                const colIdx = _activeCell.cellIndex + 1;
                for (let row of table.rows) {
                    const c = row.insertCell(colIdx);
                    c.style.cssText = _activeCell.style.cssText;
                    c.innerHTML = "";
                }
            }
            markNeuralDocsUnsaved();
            hideObjectOverlays();
        }

        function alignObj(dir) {
            if (!_manipulatingObj) return;
            if (dir === 'left') _manipulatingObj.style.margin = '10px auto 10px 0';
            else if (dir === 'center') _manipulatingObj.style.margin = '10px auto';
            else if (dir === 'right') _manipulatingObj.style.margin = '10px 0 10px auto';
            updateOverlayPos(_manipulatingObj);
            markNeuralDocsUnsaved();
        }

        function deleteObj() {
            if (_manipulatingObj) {
                _manipulatingObj.remove();
                hideObjectOverlays();
                markNeuralDocsUnsaved();
            }
        }

        function updateOverlayPos(el) {
            const overlay = document.getElementById('obj-resizer-overlay');
            const rect = el.getBoundingClientRect();
            overlay.style.top = (window.scrollY + rect.top) + 'px';
            overlay.style.left = (window.scrollX + rect.left) + 'px';
            overlay.style.width = rect.width + 'px';
            overlay.style.height = rect.height + 'px';
        }

        function hideObjectOverlays() {
            document.getElementById('obj-resizer-overlay').style.display = 'none';
            document.getElementById('obj-control-bubble').style.display = 'none';
        }
"""

# Place it after link tooltip logic
if 'setupLinkTooltip' in html:
    html = html.replace('function setupLinkTooltip() {', obj_logic + '\n        function setupLinkTooltip() {')

# Init in DOMContentLoaded
html = html.replace('setupLinkTooltip();', 'setupLinkTooltip(); initObjectManipulation();')

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: V15 Logic implemented.")
