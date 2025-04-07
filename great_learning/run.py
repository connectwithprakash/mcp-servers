#!/usr/bin/env python3
"""
Run script for Great Learning data scraper
This script will:
1. Check if Poetry is installed
2. Install dependencies
3. Check if Ollama is installed and running
4. Run the data scraper
"""

import sys
import subprocess


def check_poetry_installed():
    """Check if Poetry is installed on the system"""
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def install_dependencies():
    """Install dependencies using Poetry"""
    print("Installing dependencies with Poetry...")
    try:
        subprocess.run(["poetry", "install"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_ollama():
    """Run the Ollama check script with Poetry"""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(["poetry", "run", "python", "great_learning/check_ollama.py"], check=False)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def run_scraper():
    """Run the Great Learning data scraper with Poetry"""
    print("Running Great Learning data scraper...")
    try:
        subprocess.run(["poetry", "run", "python", "great_learning/create_data.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main function to run the entire pipeline"""
    # Check if Poetry is installed
    if not check_poetry_installed():
        print("\n❌ Poetry is not installed on this system.")
        print("\nTo install Poetry, visit: https://python-poetry.org/docs/#installation")
        print("\nYou can use:")
        print("  curl -sSL https://install.python-poetry.org | python3 -")
        return False

    print("✅ Poetry is installed.")

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies with Poetry.")
        print("\nTry manually with:")
        print("  poetry install")
        return False

    print("✅ Dependencies installed successfully.")

    # Check Ollama installation and setup
    if not check_ollama():
        print("\n❌ Ollama check failed. Please fix the issues before continuing.")
        return False

    # Run the scraper
    if not run_scraper():
        print("\n❌ Failed to run the Great Learning data scraper.")
        print("\nCheck the error messages above for details.")
        return False

    print("\n✅ Data scraping completed successfully!")
    print("\nYou can find the results in:")
    print("  - great_learning_full.txt: Raw scraped content")
    print("  - great_learning_vectorstore.parquet: Vector store for similarity search")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
