#!/usr/bin/env python3
"""
Launch script for CAIM Streamlit demo.
Run this from the project root directory.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    demo_path = project_root / "caim" / "demo" / "streamlit_app.py"
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("🚀 Launching CAIM Streamlit Demo...")
    print(f"📂 Project root: {project_root}")
    print(f"🎯 Demo file: {demo_path}")
    print("🌐 The app will open in your browser at http://localhost:8501")
    print("\n" + "="*60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()