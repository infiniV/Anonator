# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get virtual environment paths
venv_path = Path('.venv/Lib/site-packages')
torch_lib_path = venv_path / 'torch' / 'lib'
face_detection_path = venv_path / 'face_detection'
torch_hub_cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'

# Collect PyTorch CUDA DLLs (essential ones only to reduce size)
torch_binaries = []
essential_cuda_dlls = [
    'cublas64_12.dll',
    'cublasLt64_12.dll',
    'cudart64_12.dll',
    'cudnn64_9.dll',
    'cudnn_cnn64_9.dll',
    'cudnn_ops64_9.dll',
    'c10.dll',
    'c10_cuda.dll',
]

if torch_lib_path.exists():
    for dll_name in essential_cuda_dlls:
        dll_path = torch_lib_path / dll_name
        if dll_path.exists():
            torch_binaries.append((str(dll_path), '.'))
            print(f"Including: {dll_name}")

# Collect face_detection package data (configs, model definitions)
face_detection_datas = []
if face_detection_path.exists():
    # Include all Python files and configs from face_detection
    for pattern in ['**/*.py', '**/*.yaml', '**/*.json']:
        for file in face_detection_path.glob(pattern):
            rel_path = file.relative_to(venv_path)
            face_detection_datas.append((str(file), str(rel_path.parent)))
            print(f"Including: {rel_path}")

# Collect pre-trained RetinaFace model weights from torch hub cache
if torch_hub_cache.exists():
    for model_file in torch_hub_cache.glob('*'):
        if model_file.is_file():
            # Include model weights in the distribution
            face_detection_datas.append((str(model_file), 'torch_hub_checkpoints'))
            print(f"Including model: {model_file.name}")

# Hidden imports required for the application
hidden_imports = [
    'tkinter',
    'tkinter.ttk',
    'tkinterdnd2',
    'cv2',
    'numpy',
    'PIL',
    'PIL._tkinter_finder',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.hub',
    'torchvision',
    'torchvision.models',
    'torchvision.ops',
    'face_detection',
    'face_detection.retinaface',
    'face_detection.retinaface.detect',
    'face_detection.build',
    'face_detection.registry',
]

a = Analysis(
    ['src/anonator/main.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=face_detection_datas,
    hiddenimports=hidden_imports,
    hookspath=['pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks/runtime_hook_torch.py'],
    excludes=[
        'onnxruntime',
        'onnx',
        'tensorflow',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Anonator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled - causes issues with large DLLs
    console=False,  # Windowed application (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Anonator',
)
