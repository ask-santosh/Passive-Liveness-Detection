# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\ALOK\\Desktop\\Liveness_Model_Training'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

a.datas += [   ('deploy.prototxt', '.\\deploy.prototxt', 'DATA'),
               ('res10_300x300_ssd_iter_140000.caffemodel', '.\\res10_300x300_ssd_iter_140000.caffemodel', 'DATA'),
               ('liveness.model', '.\\liveness.model', 'DATA'),
               ('le.pickle', '.\\le.pickle', 'DATA'),
               ('deer_decode.jpg', '.\\deer_decode.jpg', 'DATA')   ]


exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='AirfaceTraining',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon = "C:\\Users\\ALOK\\Desktop\\Liveness_Model_Training\\icon.ico" )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='AirfaceTraining',
               icon = "C:\\Users\\ALOK\\Desktop\\Liveness_Model_Training\\icon.ico")

