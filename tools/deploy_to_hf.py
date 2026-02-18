#!/usr/bin/env python3
"""
Deploy RTMODT to Hugging Face Spaces
====================================
Automates the safe deployment process:
1. Ensures git status is clean
2. Switches to 'hf-deploy' orphan branch (no heavy history)
3. Syncs latest code from 'main'
4. Pushes to Hugging Face
5. Switches back to 'main'

Usage:
    python tools/deploy_to_hf.py
"""

import subprocess
import sys
import shutil

def run(cmd, exit_on_error=True):
    print(f"▶ {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Command failed: {cmd}")
        if exit_on_error:
            sys.exit(1)

def main():
    print("🚀 Deploying RTMODT to Hugging Face Spaces...\n")

    # 1. Check for uncommitted changes
    status = subprocess.getoutput("git status --porcelain")
    if status:
        print("❌ You have uncommitted changes. Please commit or stash them first.")
        print(status)
        sys.exit(1)

    try:
        # Get current branch
        branch = subprocess.getoutput("git branch --show-current").strip()
        if branch != "main":
            print(f"⚠️  You are on branch '{branch}'. Switching to 'main'...")
            run("git checkout main")

        # 2. Update main
        print("\n📥 Pulling latest main...")
        run("git pull")

        # 3. Switch to deployment branch
        print("\n🔀 Switching to 'hf-deploy' branch...")
        if "hf-deploy" not in subprocess.getoutput("git branch"):
            # Create orphan branch if not exists
            run("git checkout --orphan hf-deploy")
            run("git rm -rf .")
        else:
            run("git checkout hf-deploy")

        # 4. Sync content from main
        print("\n📦 Syncing files from main...")
        # Check out all files from main into working directory
        run("git checkout main -- .")
        
        # Reset index to match working directory (unstages everything first)
        run("git reset")
        
        # Add all files (respecting .gitignore)
        run("git add .")

        # 5. Commit
        print("\nabcd Committing deployment build...")
        run('git commit -m "Deploy: Sync with main"')

        # 6. Push to HF
        print("\n⬆️  Pushing to Hugging Face Spaces...")
        run("git push space hf-deploy:main --force")

        print("\n✅ Deployment successful!")
        print("   Build logs: https://huggingface.co/spaces/iamgabbarxd/rtmodt")

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # 7. cleanup - switch back to main
        print("\n🔙 Switching back to main...")
        run("git checkout main")

if __name__ == "__main__":
    main()
