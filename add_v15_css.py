import os

css_path = r"c:\Users\TheGoyFather\.gemini\antigravity\scratch\pdf_translator_prototype\new app\web_ui\style.css"

with open(css_path, "a", encoding="utf-8") as f:
    f.write("""
/* Advanced Object Manipulation (V15) */

/* Resizer Overlay */
.obj-resizer-overlay {
    position: absolute;
    border: 2px solid var(--accent-green);
    pointer-events: none;
    z-index: 9999;
    display: none;
}
.resizer-handle {
    position: absolute;
    width: 10px;
    height: 10px;
    background: #fff;
    border: 1px solid var(--accent-green);
    pointer-events: auto;
    border-radius: 2px;
}
.resizer-handle.nw { top: -6px; left: -6px; cursor: nw-resize; }
.resizer-handle.ne { top: -6px; right: -6px; cursor: ne-resize; }
.resizer-handle.sw { bottom: -6px; left: -6px; cursor: sw-resize; }
.resizer-handle.se { bottom: -6px; right: -6px; cursor: se-resize; }

/* Control Bubbles (Floating Toolbars) */
.obj-control-bubble {
    position: absolute;
    background: #1a1a1a;
    border: 1px solid rgba(74, 222, 128, 0.3);
    border-radius: 8px;
    padding: 6px;
    display: flex;
    gap: 4px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.6);
    z-index: 10001;
    pointer-events: auto;
    animation: bubbleIn 0.2s ease-out;
}
@keyframes bubbleIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.obj-btn {
    background: none;
    border: none;
    color: #acc;
    padding: 4px 8px;
    cursor: pointer;
    border-radius: 4px;
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
}
.obj-btn:hover {
    background: rgba(74, 222, 128, 0.1);
    color: var(--accent-green);
}
.obj-btn svg { width: 14px; height: 14px; }
.obj-btn.danger:hover {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
}

.tb-plus-hint {
    position: absolute;
    width: 18px;
    height: 18px;
    background: var(--accent-green);
    color: #000;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: bold;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.2s;
    z-index: 10000;
}
.tb-plus-hint:hover { opacity: 1; }
""")

print("SUCCESS: V15 CSS added.")
