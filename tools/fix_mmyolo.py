# fix_mmyolo.py
import os
import sys
import shutil

def fix_mmyolo():
    print("=" * 60)
    print("Fixing MMYOLO Installation for DJI/DroneVision")
    print("=" * 60)
    
    # Change to mmyolo directory
    mmyolo_dir = os.path.join(os.getcwd(), "mmyolo")
    if not os.path.exists(mmyolo_dir):
        print(f"❌ MMYOLO directory not found: {mmyolo_dir}")
        return False
    
    os.chdir(mmyolo_dir)
    print(f"✓ Working in: {os.getcwd()}")
    
    # 1. Backup original setup.py
    setup_file = "setup.py"
    backup_file = "setup.py.backup"
    
    if os.path.exists(setup_file):
        shutil.copy2(setup_file, backup_file)
        print(f"✓ Backed up: {setup_file} -> {backup_file}")
    else:
        print(f"❌ {setup_file} not found!")
        return False
    
    # 2. Read the file
    with open(setup_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 3. Find and replace the problematic line
    target_line = "from torch.utils.cpp_extension import BuildExtension"
    replacement = """try:
    from torch.utils.cpp_extension import BuildExtension
except ImportError:
    # Allow installation without torch (will be installed later)
    class BuildExtension:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def build_extensions(self):
            pass"""
    
    if target_line in content:
        content = content.replace(target_line, replacement)
        print("✓ Patched torch import in setup.py")
    else:
        print("⚠️  Target line not found, checking structure...")
    
    # 4. Also fix parse_requirements to skip torch check
    parse_func_start = "def parse_requirements"
    if parse_func_start in content:
        # Add import at top of function
        func_content = """def parse_requirements(fname='requirements.txt', with_version=True):
    \"\"\"Parse the package dependencies listed in a requirements file but strips
    specific versioning information.\"\"\"
    import re
    import sys
    from os.path import exists
    require_fpath = fname
    
    # Skip torch during installation
    skip_packages = ['torch', 'torchvision']
    
    def parse_line(line):
        \"\"\"Parse information from a line in a requirements text file.\"\"\"
        # Skip torch packages
        for skip in skip_packages:
            if skip in line.lower():
                return []
                
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info
    
    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)
    
    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item
    
    packages = list(gen_packages_items())
    return packages"""
        
        # Find and replace the function
        import re
        pattern = r'def parse_requirements\([^)]+\):.+?(?=\n\S|\Z)'
        content = re.sub(pattern, func_content, content, flags=re.DOTALL)
        print("✓ Patched parse_requirements function")
    
    # 5. Write the patched file
    with open(setup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Patched setup.py written")
    
    # 6. Now install
    print("\n" + "=" * 60)
    print("Installing MMYOLO...")
    print("=" * 60)
    
    # Try installation
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                          capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\n✅ MMYOLO installation SUCCESSFUL!")
        
        # Test import
        print("\nTesting import...")
        test_code = """
try:
    from mmyolo.utils import register_all_modules
    register_all_modules()
    print('✅ MMYOLO imported successfully!')
    print('✅ Ready for DJI/DroneVision development!')
except Exception as e:
    print(f'❌ Import failed: {e}')
"""
        subprocess.run([sys.executable, "-c", test_code])
        
        return True
    else:
        print("\n❌ Installation failed")
        return False

if __name__ == "__main__":
    success = fix_mmyolo()
    if not success:
        print("\n" + "=" * 60)
        print("ALTERNATIVE: Manual installation")
        print("=" * 60)
        
        # Fallback: Manual copy
        import site
        import glob
        
        site_packages = site.getsitepackages()[0]
        print(f"Site packages: {site_packages}")
        
        # Find mmyolo source
        source_dir = os.path.join(os.getcwd(), "mmyolo")
        dest_dir = os.path.join(site_packages, "mmyolo")
        
        if os.path.exists(source_dir):
            print(f"Copying {source_dir} to {dest_dir}")
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(source_dir, dest_dir)
            print("✅ Manually copied MMYOLO to site-packages")