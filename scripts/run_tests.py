"""
Test runner for SentiSight
Runs all tests with coverage
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧪 SentiSight Test Suite")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov=api",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    
    print("\nRunning tests...")
    print("Command:", " ".join(cmd))
    print("=" * 70)
    print()
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ All tests passed!")
            print("=" * 70)
            print("\nCoverage report generated in htmlcov/index.html")
        else:
            print("\n" + "=" * 70)
            print("❌ Some tests failed")
            print("=" * 70)
            sys.exit(1)
    
    except FileNotFoundError:
        print("\n❌ pytest not found. Install test dependencies:")
        print("   uv sync")
        sys.exit(1)

if __name__ == "__main__":
    main()
