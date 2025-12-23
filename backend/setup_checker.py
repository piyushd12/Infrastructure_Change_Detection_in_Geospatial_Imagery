#!/usr/bin/env python3
"""
Setup verification script for Infrastructure Damage Detection Backend
Run this before starting the server to verify all requirements are met
"""

import sys
import os
from pathlib import Path
import importlib.util

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} - OK")
        return True
    else:
        print_error(f"Python {version_str} - Need Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} NOT installed")
            all_ok = False
    
    if not all_ok:
        print(f"\n{YELLOW}Install missing packages with:{RESET}")
        print("pip install -r requirements.txt")
    
    return all_ok

def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    print_header("Checking PyTorch and CUDA")
    
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print_success(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("CUDA not available - will use CPU (slower)")
            print("  Consider installing CUDA for GPU acceleration")
            return True
    except Exception as e:
        print_error(f"Error checking PyTorch: {e}")
        return False

def check_xview2_repository():
    """Check if xView2 repository exists"""
    print_header("Checking xView2 Repository")
    
    xview2_paths = [
        './xView2_first_place',
        '../xView2_first_place',
        './xview2_first_place'
    ]
    
    found = False
    for path in xview2_paths:
        if os.path.exists(path):
            print_success(f"Found xView2 repository at: {path}")
            found = True
            
            # Check for required files
            required_files = [
                'zoo/models.py',
                'utils.py'
            ]
            
            for file in required_files:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    print_success(f"  Found: {file}")
                else:
                    print_error(f"  Missing: {file}")
                    found = False
            break
    
    if not found:
        print_error("xView2 repository not found")
        print(f"\n{YELLOW}Setup instructions:{RESET}")
        print("1. Clone the repository:")
        print("   git clone https://github.com/DIUx-xView/xView2_first_place.git")
        print("2. Or download from: https://github.com/DIUx-xView/xView2_first_place")
        return False
    
    return found

def check_model_weights():
    """Check if model weight files exist"""
    print_header("Checking Model Weights")
    
    weights_dir = 'weights'
    required_weights = [
        'res50_loc_1_tuned_best',
        'res50_cls_cce_1_0_last'
    ]
    
    if not os.path.exists(weights_dir):
        print_error(f"Weights directory not found: {weights_dir}")
        print(f"\n{YELLOW}Create weights directory and add checkpoint files:{RESET}")
        print(f"mkdir {weights_dir}")
        return False
    
    all_ok = True
    for weight_file in required_weights:
        weight_path = os.path.join(weights_dir, weight_file)
        if os.path.exists(weight_path):
            size = os.path.getsize(weight_path) / (1024 * 1024)  # MB
            print_success(f"Found: {weight_file} ({size:.1f} MB)")
        else:
            print_error(f"Missing: {weight_file}")
            all_ok = False
    
    if not all_ok:
        print(f"\n{YELLOW}Add your trained model checkpoints to the weights/ folder{RESET}")
    
    return all_ok

def check_directories():
    """Check if required directories exist"""
    print_header("Checking Directories")
    
    required_dirs = ['uploads', 'results', 'temp', 'logs']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_warning(f"Creating directory: {dir_name}/")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def check_local_utils():
    """Check if local utils.py exists"""
    print_header("Checking Local Utils")
    
    if os.path.exists('utils.py'):
        print_success("Local utils.py found")
        
        # Check if it has preprocess_inputs function
        try:
            spec = importlib.util.spec_from_file_location("utils", "utils.py")
            utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils)
            
            if hasattr(utils, 'preprocess_inputs'):
                print_success("  preprocess_inputs function available")
                return True
            else:
                print_warning("  preprocess_inputs function not found")
                return False
        except Exception as e:
            print_error(f"  Error loading utils.py: {e}")
            return False
    else:
        print_warning("Local utils.py not found (will try xView2 utils)")
        return True  # Not critical if xView2 utils exists

def check_config_files():
    """Check configuration files"""
    print_header("Checking Configuration Files")
    
    if os.path.exists('config.py'):
        print_success("config.py found")
    else:
        print_warning("config.py not found (optional)")
    
    if os.path.exists('.env'):
        print_success(".env file found")
    else:
        print_warning(".env file not found (will use defaults)")
        if os.path.exists('.env.example'):
            print("  Copy .env.example to .env to customize settings")
    
    return True

def test_import_app():
    """Try to import the Flask app"""
    print_header("Testing Flask App Import")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try importing the app
        import app
        print_success("Successfully imported app.py")
        
        # Check if models loaded
        if hasattr(app, 'loc_model') and hasattr(app, 'cls_model'):
            print_success("Model variables defined")
        
        return True
    except ImportError as e:
        print_error(f"Failed to import app.py: {e}")
        return False
    except Exception as e:
        print_error(f"Error importing app: {e}")
        return False

def run_all_checks():
    """Run all verification checks"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{'Infrastructure Damage Detection':^60}{RESET}")
    print(f"{BLUE}{'Backend Setup Verification':^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PyTorch & CUDA", check_torch_cuda),
        ("xView2 Repository", check_xview2_repository),
        ("Model Weights", check_model_weights),
        ("Directories", check_directories),
        ("Local Utils", check_local_utils),
        ("Config Files", check_config_files),
        ("Flask App", test_import_app)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{name:.<40} {status}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print(f"{GREEN}✓ All checks passed! You're ready to start the server.{RESET}")
        print(f"\n{BLUE}Start the server with:{RESET}")
        print("  python app.py")
        return True
    else:
        print(f"{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
        return False

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
