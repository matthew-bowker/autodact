# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Windows executable."""

from PyInstaller.utils.hooks import collect_all, collect_binaries, collect_submodules

# Aggressively collect spaCy/thinc/presidio — their plugin systems
# use dynamic imports that PyInstaller cannot trace statically.
spacy_datas, spacy_binaries, spacy_hiddenimports = collect_all("spacy")
thinc_datas, thinc_binaries, thinc_hiddenimports = collect_all("thinc")
presidio_datas, presidio_binaries, presidio_hiddenimports = collect_all(
    "presidio_analyzer"
)

a = Analysis(
    ["src\\main.py"],
    pathex=["."],
    datas=[
        ("assets\\icon.png", "assets"),
        ("src\\pipeline\\data\\*.txt", "src\\pipeline\\data"),
        *spacy_datas,
        *thinc_datas,
        *presidio_datas,
    ],
    binaries=[
        *spacy_binaries,
        *thinc_binaries,
        *presidio_binaries,
        *collect_binaries("llama_cpp"),
    ],
    hiddenimports=[
        "PyQt6.QtSvg",
        "PyQt6.QtSvgWidgets",
        "spacy.lang.en",
        "json_repair",
        "huggingface_hub",
        "openpyxl",
        "docx",
        "llama_cpp",
        *spacy_hiddenimports,
        *thinc_hiddenimports,
        *presidio_hiddenimports,
        *collect_submodules("llama_cpp"),
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
    icon="assets\\icon.ico",
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
