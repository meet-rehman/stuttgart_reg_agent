import os
import sys
import subprocess

def check_tesseract():
    """Check if Tesseract is installed and accessible"""
    
    # Common installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    
    tesseract_path = None
    
    # Check if tesseract is in PATH
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract found in PATH")
            tesseract_path = 'tesseract'
    except FileNotFoundError:
        print("⚠️  Tesseract not found in PATH")
    
    # Check common installation paths
    if not tesseract_path:
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Tesseract found at: {path}")
                tesseract_path = path
                break
    
    if not tesseract_path:
        print("❌ Tesseract not found!")
        print("Please install Tesseract from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    
    return tesseract_path

def check_german_language(tesseract_path):
    """Check if German language pack is installed"""
    
    if tesseract_path == 'tesseract':
        cmd = ['tesseract', '--list-langs']
    else:
        cmd = [tesseract_path, '--list-langs']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        languages = result.stdout
        
        if 'deu' in languages:
            print("✅ German language pack (deu) is installed")
            return True
        else:
            print("⚠️  German language pack (deu) NOT found")
            print("Available languages:", languages)
            return False
    except Exception as e:
        print(f"❌ Error checking languages: {e}")
        return False

def check_poppler():
    """Check if Poppler is installed"""
    
    try:
        result = subprocess.run(['pdftoppm', '-v'], 
                              capture_output=True, text=True, 
                              stderr=subprocess.STDOUT)
        print("✅ Poppler is installed")
        return True
    except FileNotFoundError:
        print("⚠️  Poppler not found")
        print("Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("Or install via: pip install poppler-windows")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    
    required = {
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image', 
        'PIL': 'Pillow'
    }
    missing = []
    
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing)}")
    
    return len(missing) == 0

def main():
    print("="*60)
    print("🔍 OCR Setup Diagnostic Check")
    print("="*60)
    print()
    
    # Check Tesseract
    print("1️⃣  Checking Tesseract...")
    tesseract_path = check_tesseract()
    print()
    
    # Check German language
    if tesseract_path:
        print("2️⃣  Checking German language pack...")
        has_german = check_german_language(tesseract_path)
        print()
    
    # Check Poppler
    print("3️⃣  Checking Poppler (PDF converter)...")
    has_poppler = check_poppler()
    print()
    
    # Check Python packages
    print("4️⃣  Checking Python packages...")
    has_packages = check_python_packages()
    print()
    
    print("="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if tesseract_path:
        print(f"✅ Tesseract: FOUND at {tesseract_path}")
        print(f"\n💡 Add this to your Python code:")
        print(f"   pytesseract.pytesseract.tesseract_cmd = r'{tesseract_path}'")
    else:
        print("❌ Tesseract: NOT FOUND")
        print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    print()
    
    if tesseract_path and has_german:
        print("✅ German language: READY")
    elif tesseract_path:
        print("⚠️  German language: MISSING")
        print("   Reinstall Tesseract and select 'German' during installation")
    
    print()
    
    if has_poppler:
        print("✅ Poppler: READY")
    else:
        print("⚠️  Poppler: MISSING")
    
    print()
    
    if has_packages:
        print("✅ Python packages: ALL INSTALLED")
    else:
        print("⚠️  Python packages: SOME MISSING (see above)")
    
    print()
    print("="*60)

if __name__ == "__main__":
    main()