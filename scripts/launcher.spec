# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Anonator Launcher."""

import sys
from pathlib import Path

block_cipher = None

# Get project root (parent of scripts/)
project_root = Path(SPECPATH).parent

# Launcher files
launcher_dir = project_root / 'launcher'

a = Analysis(
    [str(project_root / 'launcher' / 'launcher_main.py')],  # Entry point
    pathex=[],
    binaries=[],
    datas=[
        # Include launcher package
        (str(launcher_dir), 'launcher'),
    ],
    hiddenimports=[
        'customtkinter',
        'PIL._tkinter_finder',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy dependencies not needed for launcher
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'mediapipe',
        'insightface',
        'onnxruntime',
        'ultralytics',
    ],
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
    name='AnonatorLauncher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable
    console=False,  # Windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'resources' / 'logo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AnonatorLauncher',
)
