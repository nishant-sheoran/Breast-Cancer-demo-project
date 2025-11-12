"""
Automated deployment script for Breast Cancer Detection Streamlit App to Hugging Face Spaces.

This script:
1. Checks for required files (streamlit_app.py, requirements.txt, README.md)
2. Ensures model.pkl exists (copies from backend if needed)
3. Initializes Git repo if not exists
4. Provides instructions for Hugging Face deployment
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def print_step(step_num: int, message: str):
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {message}")
    print(f"{'='*60}\n")


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def ensure_model_file():
    """Ensure model.pkl exists in root directory."""
    model_root = Path("model.pkl")
    model_backend = Path("backend/breast_cancer_model.pkl")
    
    if model_root.exists():
        print("✅ model.pkl already exists in root directory")
        return True
    
    if model_backend.exists():
        print(f"📋 Copying {model_backend} to model.pkl...")
        shutil.copy2(model_backend, model_root)
        print("✅ model.pkl created successfully")
        return True
    
    print("⚠️  WARNING: No model.pkl found in root or backend directory")
    print("   The Streamlit app will require users to upload a model file.")
    return False


def check_requirements_txt():
    """Check and enhance requirements.txt if needed."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found — creating one...")
        reqs = textwrap.dedent("""
            streamlit>=1.36
            scikit-learn>=1.4
            pandas>=2.1
            numpy>=1.26
            shap>=0.45
            matplotlib>=3.8
        """).strip()
        with open("requirements.txt", "w") as f:
            f.write(reqs)
        print("✅ requirements.txt created")
        return False
    
    # Check if all essential dependencies are present
    with open("requirements.txt", "r") as f:
        content = f.read().lower()
    
    essential = ["streamlit", "scikit-learn", "pandas", "numpy"]
    missing = [dep for dep in essential if dep not in content]
    
    if missing:
        print(f"⚠️  Missing dependencies in requirements.txt: {', '.join(missing)}")
        print("   Consider adding them manually or the deployment may fail.")
    
    print("✅ requirements.txt exists and looks good")
    return True


def ensure_readme():
    """Check if README.md exists and is appropriate for HF Spaces."""
    if not os.path.exists("README.md"):
        print("❌ README.md not found — creating one...")
        readme = textwrap.dedent("""
            # 🎗️ Breast Cancer Detection Streamlit App

            This project is a machine learning-based application for breast cancer detection.
            It provides a simple Streamlit interface to input diagnostic measurements and get predictions from a trained model.

            ## Features

            - Interactive feature input form
            - Real-time prediction with probability scores
            - SHAP-based model explanations
            - Visual waterfall plots for feature contributions

            ## Deployment

            This app is deployed for free on Hugging Face Spaces using Streamlit.

            ## Usage

            Enter the 10 diagnostic measurements:
            - Mean radius
            - Mean texture
            - Mean perimeter
            - Mean area
            - Mean smoothness
            - Mean compactness
            - Mean concavity
            - Mean concave points
            - Mean symmetry
            - Mean fractal dimension

            Click "Predict" to get the model's prediction and explanation.

            ## Note

            This interface is for research support only and is **not** a medical diagnostic device.
        """).strip()
        with open("README.md", "w") as f:
            f.write(readme)
        print("✅ README.md created")
        return False
    
    print("✅ README.md exists")
    return True


def init_git_repo():
    """Initialize Git repo if it doesn't exist."""
    if os.path.exists(".git"):
        print("✅ Git repository already initialized")
        return True
    
    print("📦 Initializing Git repository...")
    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        print("✅ Git repository initialized")
        
        # Create .gitignore if it doesn't exist
        if not os.path.exists(".gitignore"):
            gitignore_content = textwrap.dedent("""
                __pycache__/
                *.pyc
                *.pyo
                *.pyd
                .Python
                .venv/
                venv/
                env/
                .env
                *.pkl
                !model.pkl
                .DS_Store
                *.log
                node_modules/
            """).strip()
            with open(".gitignore", "w") as f:
                f.write(gitignore_content)
            print("✅ .gitignore created")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize Git: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git is not installed. Please install Git first.")
        return False


def check_git_status():
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Consider committing them first.")
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    """Main deployment preparation function."""
    print("\n" + "="*60)
    print("🚀 Hugging Face Spaces Deployment Preparation")
    print("="*60)
    
    # Step 1: Check project structure
    print_step(1, "Checking Project Structure")
    
    files_ok = True
    files_ok &= check_file_exists("streamlit_app.py", "Streamlit app")
    files_ok &= check_requirements_txt()
    files_ok &= ensure_readme()
    files_ok &= ensure_model_file()
    
    if not files_ok:
        print("\n⚠️  Some files were missing and have been created.")
        print("   Please review them before proceeding.\n")
    
    # Step 2: Initialize Git
    print_step(2, "Git Repository Setup")
    git_ok = init_git_repo()
    
    if git_ok:
        check_git_status()
    
    # Step 3: Deployment instructions
    print_step(3, "Deployment Instructions")
    
    print(textwrap.dedent("""
        📋 Next Steps to Deploy to Hugging Face Spaces:
        
        1. Login to Hugging Face:
           huggingface-cli login
           (Get your token from: https://huggingface.co/settings/tokens)
        
        2. Create a new Space on Hugging Face:
           - Go to: https://huggingface.co/new-space
           - Choose a name (e.g., "breast-cancer-detector")
           - Select "Streamlit" as the SDK
           - Set visibility (Public/Private)
           - Click "Create Space"
        
        3. Add the Hugging Face remote and push:
           git remote add origin https://huggingface.co/spaces/<username>/<space-name>
           git branch -M main
           git add .
           git commit -m "Initial commit for Hugging Face deployment"
           git push -u origin main
        
        ⚠️  Important Notes:
        - Replace <username> with your Hugging Face username
        - Replace <space-name> with your Space name
        - Make sure model.pkl is committed (it should be if it exists)
        - The app will be live in ~2-5 minutes after pushing
        
        🔗 Your app will be available at:
        https://huggingface.co/spaces/<username>/<space-name>
    """))
    
    print("\n" + "="*60)
    print("✅ Deployment preparation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

