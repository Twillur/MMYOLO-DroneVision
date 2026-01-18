# setup_dji_env.py
import os
import sys

# MUST be set BEFORE any mm* imports
os.environ['MMCV_WITH_OPS'] = '0'

print("Setting up DJI MMYOLO environment...")

# Now import
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import mmcv
    print(f"✅ MMCV-lite: {mmcv.__version__}")
    
    # Patch mmcv to avoid ops loading
    import mmcv.utils.ext_loader as ext_loader
    original_load_ext = ext_loader.load_ext
    
    def patched_load_ext(name):
        if '_ext' in name:
            raise ImportError(f"mmcv ops disabled: {name}")
        return original_load_ext(name)
    
    ext_loader.load_ext = patched_load_ext
    print("✅ Patched mmcv ops loader")
    
    # Now try MMDet and MMYOLO
    import mmdet
    print(f"✅ MMDet: {mmdet.__version__}")
    
    from mmyolo.utils import register_all_modules
    register_all_modules()
    print("✅ MMYOLO modules registered!")
    
    print("\n" + "=" * 60)
    print("🎉 DJI MMYOLO ENVIRONMENT READY!")
    print("=" * 60)
    
    # Save environment info
    with open('dji_env_ready.txt', 'w') as f:
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"MMCV-lite: {mmcv.__version__}\n")
        f.write(f"MMDet: {mmdet.__version__}\n")
        f.write("MMYOLO: Registered\n")
        f.write("Environment: Ready for DJI-style development\n")
    
    print("\n✅ Environment info saved to: dji_env_ready.txt")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()