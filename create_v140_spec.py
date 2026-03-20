import os

spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        ('bin/tesseract/tesseract.exe', 'bin/tesseract'),
        ('bin/tesseract/*.dll', 'bin/tesseract'),
        ('bin/tesseract/tessdata/*', 'bin/tesseract/tessdata')
    ],
    datas=[],
    hiddenimports=[
        'customtkinter',
        'fitz',
        'PIL',
        'pytesseract',
        'deep_translator',
        'deepl',
        'openai',
        'Supabase',
        'gotrue',
        'postgrest',
        'storage3',
        'realtime',
        'httpx',
        'asyncio',
        'cv2',
        'numpy',
        'urllib.request',
        'docx',
        'ebooklib',
        'ebooklib.epub',
        'bs4',
        'reportlab',
        'reportlab.pdfgen.canvas',
        'reportlab.lib.pagesizes',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BilingualBookTranslatorPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,

)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BilingualBookTranslatorPro_v142',
)
"""

with open('BilingualBookTranslatorPro_v142.spec', 'w') as f:
    f.write(spec_content)

print("Created BilingualBookTranslatorPro_v142.spec")
