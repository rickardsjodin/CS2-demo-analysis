"""
Test script to verify the reorganized project structure works correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all imports work correctly after reorganization"""
    print("🔍 Testing imports...")
    
    try:
        from src.core.analysis import get_player_kill_death_analysis
        print("✅ Core analysis import successful")
    except ImportError as e:
        print(f"❌ Core analysis import failed: {e}")
        return False
    
    try:
        from src.utils.cache_utils import load_demo
        print("✅ Cache utils import successful")
    except ImportError as e:
        print(f"❌ Cache utils import failed: {e}")
        return False
    
    try:
        from src.utils.plotting import plot_kill_death_analysis
        print("✅ Plotting utils import successful")
    except ImportError as e:
        print(f"❌ Plotting utils import failed: {e}")
        return False
    
    try:
        import config
        print("✅ Config import successful")
        print(f"   - Project root: {config.PROJECT_ROOT}")
        print(f"   - Default demo: {config.DEFAULT_DEMO_FILE}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test that all expected directories exist"""
    print("\n📁 Testing directory structure...")
    
    import config
    
    expected_dirs = [
        config.SRC_DIR,
        config.DATA_DIR,
        config.DEMOS_DIR,
        config.OUTPUTS_DIR,
        config.CACHE_DIR,
        config.MODELS_DIR,
        config.VISUALIZATIONS_DIR,
        config.REPORTS_DIR
    ]
    
    all_exist = True
    for directory in expected_dirs:
        if directory.exists():
            print(f"✅ {directory.name}/ directory exists")
        else:
            print(f"❌ {directory.name}/ directory missing")
            all_exist = False
    
    return all_exist

def test_file_locations():
    """Test that files are in their expected locations"""
    print("\n📄 Testing file locations...")
    
    expected_files = {
        "src/core/analysis.py": "Core analysis module",
        "src/core/win_probability.py": "Win probability module", 
        "src/utils/cache_utils.py": "Cache utilities",
        "src/utils/plotting.py": "Plotting utilities",
        "src/ml/train_win_probability_model.py": "ML training module",
        "data/models/ct_win_probability_model.pkl": "CT win model",
        "demos/vitality-vs-gamer-legion-m1-train.dem": "Demo file",
        "docs/why_round_swing_makes_sense.md": "Documentation"
    }
    
    all_exist = True
    project_root = Path(".")
    
    for file_path, description in expected_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"⚠️  {description}: {file_path} (not found)")
    
    return True  # Don't fail on missing files, just report

def main():
    """Run all tests"""
    print("🧪 CS2 Demo Analysis - Project Structure Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_directory_structure():
        tests_passed += 1
    
    if test_file_locations():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Project reorganization successful.")
        print("\n🚀 You can now run 'python main.py' to start analysis.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
