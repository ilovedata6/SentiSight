"""
Launch script for SentiSight Streamlit Dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🎯 SentiSight - Streamlit Dashboard")
    print("=" * 70)
    print("\nStarting dashboard...")
    print("Once loaded, the dashboard will open in your browser")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 70)
    
    # Path to streamlit app
    app_path = Path(__file__).parent.parent / "frontend" / "app.py"
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  uv sync")

if __name__ == "__main__":
    main()
