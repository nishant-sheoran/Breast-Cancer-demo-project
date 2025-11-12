# 🚀 Deployment Guide: Hugging Face Spaces

This guide will help you deploy your Breast Cancer Detection Streamlit app to Hugging Face Spaces for free.

## Quick Start

### Option 1: Automated Preparation (Recommended)

Run the deployment preparation script:

```bash
python deploy_to_hf.py
```

This script will:
- ✅ Check for required files (`streamlit_app.py`, `requirements.txt`, `README.md`)
- ✅ Ensure `model.pkl` exists (copies from `backend/` if needed)
- ✅ Initialize Git repository if not exists
- ✅ Create `.gitignore` if needed
- ✅ Provide step-by-step deployment instructions

### Option 2: Fully Automated Deployment

For a more automated experience (requires Hugging Face CLI):

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face
huggingface-cli login

# Run automated deployment
python deploy_to_hf_auto.py --auto-push --space-name breast-cancer-detector --username YOUR_USERNAME
```

Or run interactively:
```bash
python deploy_to_hf_auto.py --auto-push
```

## Manual Deployment Steps

If you prefer to deploy manually:

### 1. Prepare Your Project

```bash
python deploy_to_hf.py
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Create a New Space

1. Go to: https://huggingface.co/new-space
2. Choose a name (e.g., `breast-cancer-detector`)
3. Select **Streamlit** as the SDK
4. Set visibility (Public/Private)
5. Click **Create Space**

### 4. Push Your Code

```bash
# Add Hugging Face remote (replace <username> and <space-name>)
git remote add origin https://huggingface.co/spaces/<username>/<space-name>

# Ensure you're on main branch
git branch -M main

# Stage all files
git add .

# Commit
git commit -m "Deploy to Hugging Face Spaces"

# Push to Hugging Face
git push -u origin main
```

### 5. Wait for Deployment

Your app will be live in ~2-5 minutes at:
```
https://huggingface.co/spaces/<username>/<space-name>
```

## Required Files

The deployment script ensures you have:

- ✅ `streamlit_app.py` - Your Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `model.pkl` - Trained model file (copied from `backend/` if needed)

## Troubleshooting

### Model File Not Found

If `model.pkl` is not in the root directory, the script will automatically copy it from `backend/breast_cancer_model.pkl`. If neither exists, users will need to upload the model through the Streamlit interface.

### Git Issues

If Git is not initialized:
- The script will automatically initialize it
- Make sure Git is installed on your system

### Hugging Face CLI Not Found

Install it with:
```bash
pip install huggingface_hub[cli]
```

### Push Fails

If automatic push fails:
1. Check your Hugging Face token is valid
2. Ensure the Space exists on Hugging Face
3. Verify your Git remote URL is correct
4. Try pushing manually: `git push -u origin main`

## Notes

- The `.gitignore` file is automatically created to exclude unnecessary files
- `model.pkl` is explicitly included in Git (needed for deployment)
- The app will automatically restart when you push new changes
- Hugging Face Spaces provides free hosting with automatic SSL certificates

## Support

For issues with:
- **Deployment scripts**: Check the script output for error messages
- **Hugging Face Spaces**: See [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- **Streamlit**: See [Streamlit Documentation](https://docs.streamlit.io/)

