# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for macOS .app bundle."""

from PyInstaller.utils.hooks import collect_all

# Aggressively collect spaCy/thinc/presidio — their plugin systems
# use dynamic imports that PyInstaller cannot trace statically.
spacy_datas, spacy_binaries, spacy_hiddenimports = collect_all("spacy")
thinc_datas, thinc_binaries, thinc_hiddenimports = collect_all("thinc")
presidio_datas, presidio_binaries, presidio_hiddenimports = collect_all(
    "presidio_analyzer"
)
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all(
    "transformers"
)
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
tokenizers_datas, tokenizers_binaries, tokenizers_hiddenimports = collect_all(
    "tokenizers"
)

a = Analysis(
    ["src/main.py"],
    pathex=["."],
    datas=[
        ("assets/icon.png", "assets"),
        ("src/pipeline/data/*.txt", "src/pipeline/data"),
        *spacy_datas,
        *thinc_datas,
        *presidio_datas,
        *transformers_datas,
        *torch_datas,
        *tokenizers_datas,
    ],
    binaries=[
        *spacy_binaries,
        *thinc_binaries,
        *presidio_binaries,
        *transformers_binaries,
        *torch_binaries,
        *tokenizers_binaries,
    ],
    hiddenimports=[
        "PyQt6.QtSvg",
        "PyQt6.QtSvgWidgets",
        "spacy.lang.en",
        "huggingface_hub",
        "openpyxl",
        "docx",
        "transformers",
        "torch",
        "tokenizers",
        *spacy_hiddenimports,
        *thinc_hiddenimports,
        *presidio_hiddenimports,
        *transformers_hiddenimports,
        *torch_hiddenimports,
        *tokenizers_hiddenimports,
    ],
    excludes=[
        "torchvision",
        "torchaudio",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Autodact",
    icon="assets/icon.icns",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Autodact",
)

app = BUNDLE(
    coll,
    name="Autodact.app",
    icon="assets/icon.icns",
    bundle_identifier="com.autodact.app",
    info_plist={
        "CFBundleShortVersionString": "1.0.1",
        "NSHighResolutionCapable": True,
        "NSAppSleepDisabled": True,
    },
)
