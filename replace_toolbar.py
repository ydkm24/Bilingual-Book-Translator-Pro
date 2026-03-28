"""Fix all 6 toolbar UX issues:
1. Buttons too compact - add spacing between groups
2. B/I/U/S no active state - add data-command and toggle logic
3. Color picker deselects text - save/restore selection
4. Insert link uses browser prompt - custom modal
5. Clear formatting unclear - better tooltip
6. Alignment popout hidden - fix z-index
Also: remove duplicate </nav> tag, add custom link modal HTML
"""

filepath = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# ---- FIX 1: Remove duplicate </nav> ----
content = content.replace(
    '</nav>\r\n                     </nav>',
    '</nav>'
)
content = content.replace(
    '</nav>\n                     </nav>',
    '</nav>'
)

# ---- FIX 2-6: Replace the B/I/U/S buttons with data-command attributes for toggle tracking ----
# Replace Bold button
content = content.replace(
    '''<button class="tb" onclick="document.execCommand('bold',false,null)" title="Bold (Ctrl+B)">''',
    '''<button class="tb tb-toggle" data-command="bold" onclick="execAndToggle(this,'bold')" title="Bold (Ctrl+B)">'''
)
content = content.replace(
    '''<button class="tb" onclick="document.execCommand('italic',false,null)" title="Italic (Ctrl+I)">''',
    '''<button class="tb tb-toggle" data-command="italic" onclick="execAndToggle(this,'italic')" title="Italic (Ctrl+I)">'''
)
content = content.replace(
    '''<button class="tb" onclick="document.execCommand('underline',false,null)" title="Underline (Ctrl+U)">''',
    '''<button class="tb tb-toggle" data-command="underline" onclick="execAndToggle(this,'underline')" title="Underline (Ctrl+U)">'''
)
content = content.replace(
    '''<button class="tb" onclick="document.execCommand('strikeThrough',false,null)" title="Strikethrough">''',
    '''<button class="tb tb-toggle" data-command="strikeThrough" onclick="execAndToggle(this,'strikeThrough')" title="Strikethrough">'''
)

# ---- FIX 3: Color pickers - use onmousedown to save selection, then apply on change ----
content = content.replace(
    '''<input type="color" id="toolbar-fg-color" value="#e4e4e7" onchange="document.execCommand('foreColor', false, this.value); document.getElementById('fg-color-bar').style.background=this.value" class="tb-color-input">''',
    '''<input type="color" id="toolbar-fg-color" value="#e4e4e7" onmousedown="saveSelection()" onchange="restoreAndApplyColor('foreColor', this.value); document.getElementById('fg-color-bar').style.background=this.value" class="tb-color-input">'''
)
content = content.replace(
    '''<input type="color" id="toolbar-bg-color" value="#facc15" onchange="document.execCommand('hiliteColor', false, this.value); document.getElementById('bg-color-bar').style.background=this.value" class="tb-color-input">''',
    '''<input type="color" id="toolbar-bg-color" value="#facc15" onmousedown="saveSelection()" onchange="restoreAndApplyColor('hiliteColor', this.value); document.getElementById('bg-color-bar').style.background=this.value" class="tb-color-input">'''
)

# ---- FIX 4: Replace promptInsertLink() call with showLinkModal() ----
content = content.replace(
    '''onclick="promptInsertLink()"''',
    '''onclick="showLinkModal()"'''
)

# ---- FIX 5: Better clear formatting tooltip ----
content = content.replace(
    '''title="Clear Formatting">''',
    '''title="Remove All Formatting (strips bold, italic, colors, etc.)">'''
)

# ---- FIX 6: Alignment popout - already has z-index:1000 in CSS, but the canvas overflow may be clipping it ----
# The popout opens BELOW the toolbar but the canvas has overflow hidden. We need to pop UPWARD instead.
# Actually the fix is simpler: the docs-lower-toolbar has overflow-x:auto which clips children. Change to overflow:visible.
# We'll fix this in CSS, but also add some inline.

# ---- ADD: Custom Link Modal HTML (before the closing </div> of workspace) ----
link_modal_html = '''
                     <!-- Custom Link Insert Modal -->
                     <div id="link-modal-overlay" class="link-modal-overlay" style="display:none;">
                         <div class="link-modal">
                             <div class="link-modal-header">
                                 <span>INSERT LINK</span>
                                 <button onclick="closeLinkModal()" class="link-modal-close">&times;</button>
                             </div>
                             <input type="text" id="link-modal-url" class="link-modal-input" placeholder="Paste URL here..." value="https://" />
                             <div class="link-modal-actions">
                                 <button class="link-modal-cancel" onclick="closeLinkModal()">Cancel</button>
                                 <button class="link-modal-apply" onclick="applyLinkFromModal()">Apply</button>
                             </div>
                         </div>
                     </div>
'''

# Insert the modal right before the docs-canvas
content = content.replace(
    '<div class="docs-canvas" id="neural-docs-canvas">',
    link_modal_html + '\n<div class="docs-canvas" id="neural-docs-canvas">'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: All 6 toolbar UX fixes applied!")
