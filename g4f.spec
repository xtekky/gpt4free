# -*- mode: python ; coding: utf-8 -*-
"""
Cross-platform PyInstaller spec for g4f
Supports Windows, Linux, and macOS builds
"""

import os
import sys
from pathlib import Path

# Get the platform
platform = sys.platform
if platform.startswith('win'):
    platform_name = 'windows'
    exe_extension = '.exe'
elif platform.startswith('linux'):
    platform_name = 'linux'  
    exe_extension = ''
elif platform.startswith('darwin'):
    platform_name = 'macos'
    exe_extension = ''
else:
    platform_name = 'unknown'
    exe_extension = ''

# Get version from environment or use default
version = os.environ.get('G4F_VERSION', '0.0.0-dev')

# Base directory
base_dir = Path(__file__).parent

# Entry point
entry_script = base_dir / 'g4f' / 'cli.py'

# Icon path (use Windows icon if available, otherwise None)
icon_path = base_dir / 'projects' / 'windows' / 'icon.ico'
icon = str(icon_path) if icon_path.exists() else None

# Data files to include
datas = []

# Include GUI files if they exist
gui_client = base_dir / 'g4f' / 'gui' / 'client'
if gui_client.exists():
    datas.append((str(gui_client), 'g4f/gui/client'))

gui_server = base_dir / 'g4f' / 'gui' / 'server'  
if gui_server.exists():
    datas.append((str(gui_server), 'g4f/gui/server'))

# Include provider data
provider_dirs = ['npm', 'gigachat_crt', 'you', 'har']
for provider_dir in provider_dirs:
    provider_path = base_dir / 'g4f' / 'Provider' / provider_dir
    if provider_path.exists():
        datas.append((str(provider_path), f'g4f/Provider/{provider_dir}'))

# Hidden imports that may be needed
hiddenimports = [
    'g4f',
    'g4f.cli',
    'g4f.client',
    'g4f.Provider',
    'g4f.gui',
    'aiohttp',
    'requests',
    'brotli',
    'pycryptodome',
    'nest_asyncio',
]

# Platform-specific adjustments
if platform_name == 'windows':
    # Windows-specific hidden imports
    hiddenimports.extend([
        'win32api',
        'win32con',
        'win32event',
        'win32file',
        'pywintypes',
    ])
elif platform_name == 'linux':
    # Linux-specific hidden imports  
    hiddenimports.extend([
        'tkinter',
    ])

# Exclude modules we don't need for CLI
excludes = [
    'tkinter',
    'matplotlib',
    'pandas',
    'numpy',
    'scipy',
    'PIL',
    'pyqt5',
    'pyqt6',
    'pyside2',
    'pyside6',
]

# Build analysis
a = Analysis(
    [str(entry_script)],
    pathex=[str(base_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executable configuration
exe_name = f'g4f-{platform_name}-{version}'
if platform_name == 'windows':
    exe_name += '.exe'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

# For macOS, create an app bundle
if platform_name == 'macos':
    app = BUNDLE(
        exe,
        name=f'g4f-{version}.app',
        icon=icon,
        bundle_identifier='ai.g4f.app',
        info_plist={
            'CFBundleDisplayName': 'g4f',
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )