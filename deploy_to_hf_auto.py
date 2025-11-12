"""
Enhanced automated deployment script with optional automatic Git push.

This script extends deploy_to_hf.py with the ability to automatically:
- Stage and commit files
- Add Hugging Face remote
- Push to Hugging Face Spaces

Usage:
    python deploy_to_hf_auto.py [--auto-push] [--space-name NAME] [--username USERNAME]
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Import functions from the base script
from deploy_to_hf import (
    check_file_exists,
    check_requirements_txt,
    ensure_model_file,
    ensure_readme,
    init_git_repo,
    print_step,
)


def check_hf_cli_installed():
    """Check if huggingface-cli is installed."""
    try:
        subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def login_to_hf():
    """Attempt to login to Hugging Face."""
    if not check_hf_cli_installed():
        print("❌ huggingface-cli is not installed.")
        print("   Install it with: pip install huggingface_hub[cli]")
        return False
    
    print("🔐 Attempting to login to Hugging Face...")
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            check=True
        )
        username = result.stdout.strip()
        print(f"✅ Already logged in as: {username}")
        return username
    except subprocess.CalledProcessError:
        print("⚠️  Not logged in. Please run: huggingface-cli login")
        return None


def create_hf_space(space_name: str, username: str):
    """Create a Hugging Face Space using the API."""
    print(f"📦 Creating Hugging Face Space: {username}/{space_name}...")
    try:
        from huggingface_hub import create_repo
        
        repo_id = f"{username}/{space_name}"
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True
        )
        print(f"✅ Space created/verified: {repo_id}")
        return True
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  Could not create space automatically: {e}")
        print("   Please create it manually at: https://huggingface.co/new-space")
        return False


def setup_git_remote(username: str, space_name: str):
    """Set up Git remote for Hugging Face Space."""
    remote_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    
    # Check if remote already exists
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True
        )
        if remote_url in result.stdout:
            print(f"✅ Git remote already set to: {remote_url}")
            return True
        else:
            print(f"⚠️  Remote 'origin' exists but points to different URL")
            print(f"   Current: {result.stdout.strip()}")
            print(f"   Expected: {remote_url}")
            response = input("   Replace it? (y/n): ").strip().lower()
            if response == 'y':
                subprocess.run(["git", "remote", "remove", "origin"], check=True)
            else:
                return False
    except subprocess.CalledProcessError:
        pass  # Remote doesn't exist, which is fine
    
    # Add the remote
    try:
        subprocess.run(
            ["git", "remote", "add", "origin", remote_url],
            check=True,
            capture_output=True
        )
        print(f"✅ Git remote added: {remote_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to add Git remote: {e}")
        return False


def commit_and_push(auto_commit: bool = True):
    """Stage, commit, and push changes to Hugging Face."""
    # Check if we're on main branch
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = result.stdout.strip()
        if current_branch != "main":
            print(f"📋 Switching to 'main' branch (currently on '{current_branch}')...")
            subprocess.run(["git", "branch", "-M", "main"], check=True)
    except subprocess.CalledProcessError:
        # No branch exists yet, create main
        subprocess.run(["git", "branch", "-M", "main"], check=True)
    
    # Check for changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip() and auto_commit:
        print("✅ No changes to commit")
    elif auto_commit:
        print("📝 Staging all changes...")
        subprocess.run(["git", "add", "."], check=True)
        
        print("💾 Committing changes...")
        subprocess.run(
            ["git", "commit", "-m", "Deploy to Hugging Face Spaces"],
            check=True
        )
        print("✅ Changes committed")
    
    # Push to Hugging Face
    print("🚀 Pushing to Hugging Face Spaces...")
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            check=True
        )
        print("✅ Successfully pushed to Hugging Face Spaces!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push: {e}")
        print("   You may need to push manually:")
        print("   git push -u origin main")
        return False


def main():
    """Main deployment function with optional automation."""
    parser = argparse.ArgumentParser(
        description="Deploy Streamlit app to Hugging Face Spaces"
    )
    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="Automatically push to Hugging Face after preparation"
    )
    parser.add_argument(
        "--space-name",
        type=str,
        help="Name of the Hugging Face Space (e.g., 'breast-cancer-detector')"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face username"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Hugging Face Spaces Deployment (Enhanced)")
    print("="*60)
    
    # Step 1: Check project structure
    print_step(1, "Checking Project Structure")
    
    files_ok = True
    files_ok &= check_file_exists("streamlit_app.py", "Streamlit app")
    files_ok &= check_requirements_txt()
    files_ok &= ensure_readme()
    files_ok &= ensure_model_file()
    
    # Step 2: Initialize Git
    print_step(2, "Git Repository Setup")
    git_ok = init_git_repo()
    
    if not git_ok:
        print("❌ Git setup failed. Cannot proceed with automated deployment.")
        sys.exit(1)
    
    # Step 3: Auto-push if requested
    if args.auto_push:
        print_step(3, "Automated Deployment")
        
        # Get username
        username = args.username
        if not username:
            username = login_to_hf()
            if not username:
                print("❌ Cannot proceed without Hugging Face username")
                print("   Please login first: huggingface-cli login")
                print("   Or provide username with: --username YOUR_USERNAME")
                sys.exit(1)
        
        # Get space name
        space_name = args.space_name
        if not space_name:
            space_name = input("Enter Space name (e.g., 'breast-cancer-detector'): ").strip()
            if not space_name:
                print("❌ Space name is required")
                sys.exit(1)
        
        # Create space (optional, will fail gracefully if it exists)
        create_hf_space(space_name, username)
        
        # Setup Git remote
        if not setup_git_remote(username, space_name):
            print("❌ Failed to setup Git remote")
            sys.exit(1)
        
        # Commit and push
        if commit_and_push():
            print("\n" + "="*60)
            print("🎉 Deployment Complete!")
            print("="*60)
            print(f"\n🔗 Your app will be live in ~2-5 minutes at:")
            print(f"   https://huggingface.co/spaces/{username}/{space_name}\n")
        else:
            print("\n⚠️  Deployment partially completed. Please check errors above.")
    else:
        # Step 3: Manual instructions
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
               git commit -m "Deploy to Hugging Face Spaces"
               git push -u origin main
            
            💡 Tip: Use --auto-push flag for automated deployment:
               python deploy_to_hf_auto.py --auto-push --space-name <name> --username <user>
            
            ⚠️  Important Notes:
            - Replace <username> with your Hugging Face username
            - Replace <space-name> with your Space name
            - Make sure model.pkl is committed (it should be if it exists)
            - The app will be live in ~2-5 minutes after pushing
            
            🔗 Your app will be available at:
            https://huggingface.co/spaces/<username>/<space-name>
        """))
    
    print("\n✅ Deployment preparation complete!\n")


if __name__ == "__main__":
    main()

