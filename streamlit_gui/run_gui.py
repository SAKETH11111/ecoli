#!/usr/bin/env python3
"""
Launcher script for CodonTransformer Streamlit GUI

This script sets up the environment and launches the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit GUI application"""

    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Add the parent directory to Python path so we can import CodonTransformer
    parent_dir = script_dir.parent
    sys.path.insert(0, str(parent_dir))

    # Set working directory to script directory
    os.chdir(script_dir)

    print("🧬 Starting CodonTransformer GUI...")
    print(f"   Working directory: {script_dir}")
    print(f"   Python path includes: {parent_dir}")

    # Check for virtual environment
    venv_path = parent_dir / "codon_env"
    if venv_path.exists():
        # Set up virtual environment paths
        venv_bin = venv_path / "bin"
        venv_python = venv_bin / "python"

        if venv_python.exists():
            print(f"✅ Found virtual environment: {venv_path}")
            # Update PATH to include virtual environment
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{venv_bin}:{current_path}"
            # Use virtual environment Python
            python_executable = str(venv_python)
        else:
            print("⚠️  Virtual environment found but Python executable missing")
            python_executable = sys.executable
    else:
        print("⚠️  No virtual environment found, using system Python")
        python_executable = sys.executable

    print(f"   Using Python: {python_executable}")
    print()

    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        return 1

    # Check if torch is available
    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ PyTorch available, using: {device}")
    except ImportError:
        print("❌ PyTorch not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        return 1

    print()
    print("🚀 Launching GUI...")
    print("   The application will open in your default web browser")
    print("   Press Ctrl+C to stop the server")
    print()

    # Launch streamlit
    try:
        subprocess.run([
            python_executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down CodonTransformer GUI...")
        return 0
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
