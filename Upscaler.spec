# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# SPECPATH is the directory containing this .spec file (injected by PyInstaller),
# so the build works from any checkout location.
HERE = SPECPATH

datas = [(os.path.join(HERE, 'upscaler', 'models', 'models'), 'upscaler/models/models'), (os.path.join(HERE, 'upscaler', 'models', 'arch'), 'upscaler/models/arch')]
binaries = []
hiddenimports = ['upscaler.plugins.upscalers.bicubic', 'upscaler.plugins.upscalers.dat', 'upscaler.plugins.upscalers.dcci', 'upscaler.plugins.upscalers.eggi_sire', 'upscaler.plugins.upscalers.hat_s', 'upscaler.plugins.upscalers.lanczos', 'upscaler.plugins.upscalers.nedi', 'upscaler.plugins.upscalers.omnisr', 'upscaler.plugins.upscalers.real_esrgan', 'upscaler.plugins.upscalers.sinc', 'upscaler.plugins.upscalers.swinir', 'upscaler.plugins.denoisers.bilateral', 'upscaler.plugins.denoisers.bm3d_plugin', 'upscaler.plugins.denoisers.nafnet', 'upscaler.plugins.denoisers.nl_means', 'upscaler.plugins.denoisers.scunet', 'upscaler.plugins.denoisers.wavelet', 'upscaler.plugins.adjusters.auto_color', 'upscaler.plugins.adjusters.auto_contrast', 'upscaler.plugins.adjusters.auto_levels', 'upscaler.plugins.adjusters.auto_tone', 'upscaler.plugins.adjusters.brightness', 'upscaler.plugins.adjusters.clarity', 'upscaler.plugins.adjusters.common', 'upscaler.plugins.adjusters.contrast', 'upscaler.plugins.adjusters.dehaze', 'upscaler.plugins.adjusters.dodge_burn', 'upscaler.plugins.adjusters.optics', 'upscaler.plugins.adjusters.refocus', 'upscaler.plugins.adjusters.saturation', 'upscaler.plugins.adjusters.shadows_highlights', 'upscaler.plugins.adjusters.sharpness', 'upscaler.plugins.adjusters.skin_smooth', 'upscaler.plugins.adjusters.split_tone', 'upscaler.plugins.adjusters.vibrance', 'upscaler.plugins.adjusters.white_balance', 'upscaler.plugins.colorizers.colormnet', 'upscaler.plugins.colorizers.ddcolor', 'upscaler.plugins.colorizers.deoldify', 'upscaler.models.arch.ddcolor_arch', 'upscaler.models.arch.deoldify_arch', 'spandrel', 'safetensors', 'einops']
tmp_ret = collect_all('spandrel')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [os.path.join(HERE, 'run.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.utils.tensorboard', 'tensorboard', 'matplotlib', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Upscaler',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Upscaler',
)
