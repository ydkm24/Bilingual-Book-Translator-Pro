import os

html_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\index.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Extract everything between <div class="docs-lower-toolbar"> and its closing </div>
start_tag = '<div class="docs-lower-toolbar"'
end_tag = '<!-- Find & Replace Modal -->'  # The next big marker after the toolbar

start_idx = html.find(start_tag)
if start_idx == -1: print("ERR: start tag not found")
start_inner = html.find('>', start_idx) + 1
end_idx = html.find(end_tag)

original_block = html[start_inner:end_idx]

# We will define the new perfectly grouped inner HTML
new_toolbar = """
                         <!-- Font Select -->
                         <select class="tb-select tb-select-wide" id="toolbar-font-face" onchange="document.execCommand('fontName', false, this.value)" title="Font">
                             <option value="Inter">Inter</option>
                             <option value="JetBrains Mono">Mono</option>
                             <option value="Georgia">Serif</option>
                             <option value="Arial">Arial</option>
                             <option value="Times New Roman">Times</option>
                         </select>

                         <!-- Font Size -->
                         <div class="tb-size-group">
                             <button class="tb-size-btn" onclick="changeFontSize(-1)">
                                 <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="5" y1="12" x2="19" y2="12"/></svg>
                             </button>
                             <input type="text" id="toolbar-font-size" value="3" class="tb-size-input" readonly />
                             <button class="tb-size-btn" onclick="changeFontSize(1)">
                                 <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                             </button>
                         </div>
                         <div class="tb-sep"></div>

                         <!-- Typography -->
                         <button class="tb tb-toggle" data-command="bold" onclick="execAndToggle(this,'bold')" title="Bold (Ctrl+B)"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h8a4 4 0 0 1 4 4 4 4 0 0 1-4 4H6z"/><path d="M6 12h9a4 4 0 0 1 4 4 4 4 0 0 1-4 4H6z"/></svg></button>
                         <button class="tb tb-toggle" data-command="italic" onclick="execAndToggle(this,'italic')" title="Italic (Ctrl+I)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="4" x2="10" y2="4"/><line x1="14" y1="20" x2="5" y2="20"/><line x1="15" y1="4" x2="9" y2="20"/></svg></button>
                         <button class="tb tb-toggle" data-command="underline" onclick="execAndToggle(this,'underline')" title="Underline (Ctrl+U)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3v7a6 6 0 0 0 6 6 6 6 0 0 0 6-6V3"/><line x1="4" y1="21" x2="20" y2="21"/></svg></button>
                         <button class="tb tb-toggle" data-command="strikeThrough" onclick="execAndToggle(this,'strikeThrough')" title="Strikethrough"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="12" x2="20" y2="12"/><path d="M17.5 7.5c0-2-2-4.5-5.5-4.5S6 5 6 7.5c0 2.5 2 3.5 6 5"/><path d="M12 12.5c4 1.5 6 2.5 6 5 0 2.5-2 4.5-6 4.5s-6-2-6-4.5"/></svg></button>
                         
                         <!-- Colors -->
                         <div class="tb-color-wrap" title="Text Color">
                             <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20h16"/><path d="M9.5 4L5 16h2l1.2-3h7.6L17 16h2L14.5 4z"/></svg>
                             <input type="color" id="toolbar-fg-color" value="#e4e4e7" onmousedown="saveSelection()" onchange="restoreAndApplyColor('foreColor', this.value); document.getElementById('fg-color-bar').style.background=this.value" class="tb-color-input">
                             <div class="tb-color-bar" id="fg-color-bar" style="background:#e4e4e7;"></div>
                         </div>
                         <div class="tb-color-wrap" title="Highlight Color">
                             <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
                             <input type="color" id="toolbar-bg-color" value="#facc15" onmousedown="saveSelection()" onchange="restoreAndApplyColor('hiliteColor', this.value); document.getElementById('bg-color-bar').style.background=this.value" class="tb-color-input">
                             <div class="tb-color-bar" id="bg-color-bar" style="background:#facc15;"></div>
                         </div>
                         <div class="tb-sep"></div>

                         <!-- Layout: Align, Spacing, Lists, Columns -->
                         <div class="tb-popout-wrap" id="align-popout-wrap">
                             <button class="tb" onclick="togglePopout('align-popout')" title="Alignment & Layout">
                                 <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="15" y2="12"/><line x1="3" y1="18" x2="18" y2="18"/></svg>
                                 <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="margin-left:2px;"><polyline points="6 9 12 15 18 9"/></svg>
                             </button>
                             <div class="tb-popout" id="align-popout">
                                 <div class="tb-popout-label">ALIGNMENT</div>
                                 <div class="tb-popout-row">
                                     <button class="tb" onclick="document.execCommand('justifyLeft',false,null);closePopouts()" title="Left"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="15" y2="12"/><line x1="3" y1="18" x2="18" y2="18"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('justifyCenter',false,null);closePopouts()" title="Center"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="6" y1="12" x2="18" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('justifyRight',false,null);closePopouts()" title="Right"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="9" y1="12" x2="21" y2="12"/><line x1="6" y1="18" x2="21" y2="18"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('justifyFull',false,null);closePopouts()" title="Justify"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg></button>
                                 </div>
                             </div>
                         </div>
                         
                         <div class="tb-popout-wrap" id="spacing-popout-wrap">
                             <button class="tb" onclick="togglePopout('spacing-popout')" title="Line Spacing">
                                 <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><path d="M3 8v8"/><path d="M3 8l2 2"/><path d="M3 8l-2 2"/><path d="M3 16l2-2"/><path d="M3 16l-2-2"/></svg>
                                 <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="margin-left:2px;"><polyline points="6 9 12 15 18 9"/></svg>
                             </button>
                             <div class="tb-popout" id="spacing-popout" style="width: 130px;">
                                 <div class="tb-popout-label">SPACING</div>
                                 <div class="tb-popout-row" style="flex-direction: column; align-items: flex-start; gap: 4px;">
                                     <button class="tb-dropdown-item" onclick="applyLineSpacing('1.0');closePopouts()">Single (1.0)</button>
                                     <button class="tb-dropdown-item" onclick="applyLineSpacing('1.15');closePopouts()">1.15</button>
                                     <button class="tb-dropdown-item" onclick="applyLineSpacing('1.5');closePopouts()">1.5</button>
                                     <button class="tb-dropdown-item" onclick="applyLineSpacing('2.0');closePopouts()">Double (2.0)</button>
                                 </div>
                             </div>
                         </div>

                         <div class="tb-popout-wrap" id="list-popout-wrap">
                             <button class="tb" onclick="togglePopout('list-popout')" title="Lists & Indent">
                                 <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="9" y1="6" x2="20" y2="6"/><line x1="9" y1="12" x2="20" y2="12"/><line x1="9" y1="18" x2="20" y2="18"/><circle cx="5" cy="6" r="1.5" fill="currentColor"/><circle cx="5" cy="12" r="1.5" fill="currentColor"/><circle cx="5" cy="18" r="1.5" fill="currentColor"/></svg>
                                 <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="margin-left:2px;"><polyline points="6 9 12 15 18 9"/></svg>
                             </button>
                             <div class="tb-popout" id="list-popout">
                                 <div class="tb-popout-label">LISTS & INDENT</div>
                                 <div class="tb-popout-row">
                                     <button class="tb" onclick="document.execCommand('insertUnorderedList',false,null);closePopouts()" title="Bullet List"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="9" y1="6" x2="20" y2="6"/><line x1="9" y1="12" x2="20" y2="12"/><line x1="9" y1="18" x2="20" y2="18"/><circle cx="5" cy="6" r="1.5" fill="currentColor"/><circle cx="5" cy="12" r="1.5" fill="currentColor"/><circle cx="5" cy="18" r="1.5" fill="currentColor"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('insertOrderedList',false,null);closePopouts()" title="Numbered List"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none"><text x="2" y="8" font-size="8" font-family="monospace">1</text><text x="2" y="14.5" font-size="8" font-family="monospace">2</text><text x="2" y="21" font-size="8" font-family="monospace">3</text><line x1="10" y1="6" x2="21" y2="6" stroke="currentColor" stroke-width="2"/><line x1="10" y1="12.5" x2="21" y2="12.5" stroke="currentColor" stroke-width="2"/><line x1="10" y1="19" x2="21" y2="19" stroke="currentColor" stroke-width="2"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('outdent',false,null);closePopouts()" title="Decrease Indent"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="7 8 3 12 7 16"/><line x1="21" y1="4" x2="11" y2="4"/><line x1="21" y1="10" x2="11" y2="10"/><line x1="21" y1="16" x2="11" y2="16"/><line x1="21" y1="22" x2="3" y2="22"/></svg></button>
                                     <button class="tb" onclick="document.execCommand('indent',false,null);closePopouts()" title="Increase Indent"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 8 7 12 3 16"/><line x1="21" y1="4" x2="11" y2="4"/><line x1="21" y1="10" x2="11" y2="10"/><line x1="21" y1="16" x2="11" y2="16"/><line x1="21" y1="22" x2="3" y2="22"/></svg></button>
                                 </div>
                             </div>
                         </div>
                         <button class="tb" onclick="toggleColumns()" title="Toggle Web/Print Columns (1 vs 2)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18"/><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg></button>
                         <div class="tb-sep"></div>

                         <!-- Insert / Symbols -->
                         <button class="tb" onclick="showLinkModal()" title="Insert Link"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg></button>
                         <button class="tb" onclick="promptInsertImage()" title="Insert Image"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg></button>
                         <button class="tb" onclick="insertBasicTable()" title="Insert Table (2x2)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></button>
                         
                         <div class="tb-popout-wrap" id="special-char-popout-wrap">
                             <button class="tb" onclick="togglePopout('special-char-popout')" title="Insert Special Character">
                                 <span style="font-family: serif; font-size: 14px; font-weight: bold; line-height: 1;">Ω</span>
                                 <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="margin-left:2px;"><polyline points="6 9 12 15 18 9"/></svg>
                             </button>
                             <div class="tb-popout" id="special-char-popout" style="width: 220px;">
                                 <div class="tb-popout-label">SYMBOLS</div>
                                 <div class="tb-popout-row" style="flex-wrap: wrap; justify-content: flex-start; gap: 4px;">
                                     <script>
                                        const chars = ['©', '™', '®', '°', '€', '£', '¥', '¢', '—', '–', '•', '←', '→', '↑', '↓', '✓', '✗', '∞', '≈', '≠', '±'];
                                        chars.forEach(c => document.write(`<button class="tb" style="width: 28px; height: 28px; justify-content:center;" onclick="document.execCommand('insertText', false, '${c}');closePopouts()">${c}</button>`));
                                     </script>
                                 </div>
                             </div>
                         </div>
                         <div class="tb-sep"></div>

                         <!-- Tools / Format -->
                         <button class="tb" onclick="document.execCommand('removeFormat',false,null)" title="Remove All Formatting"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/><line x1="3" y1="21" x2="21" y2="3" stroke="#ef4444" stroke-width="2"/></svg></button>
                         <button class="tb tb-toggle" id="btn-spellcheck" onclick="toggleSpellCheck()" title="Toggle Spell Check"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m11 17 2 2a1 1 0 1 0 3-3"/><path d="m14 14 2.5 2.5a1 1 0 1 0 3-3l-3.88-3.88a3 3 0 0 0-4.24 0l-4.24 4.24a3 3 0 0 0 0 4.24L8 19"/><path d="m2 16 1-1-1-1 4-4 4 4"/></svg></button>
                         <button class="tb" onclick="showFindReplaceModal()" title="Find & Replace"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button>

                         <!-- Spacer -->
                         <div style="flex:1;"></div>

                         <!-- AI WAND -->
                         <button class="tb-wand" onclick="triggerAIMagicWand()" title="AI Magic Wand">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
                            <span>AI WAND</span>
                         </button>
                     </div>
"""

final_html = html[:start_inner] + "\n" + new_toolbar + "\n                     \n" + html[end_idx:]

with open(html_path, "w", encoding="utf-8") as f:
    f.write(final_html)

print("SUCCESS: Toolbar re-grouped logically.")
