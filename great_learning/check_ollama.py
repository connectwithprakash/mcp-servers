import subprocess
import sys
import requests


def check_ollama_installed():
    """Check if Ollama is installed on the system"""
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_gemma_available():
    """Check if Gemma model is available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"] == "gemma" for model in models)
        return False
    except requests.exceptions.ConnectionError:
        return False


def pull_gemma_model():
    """Pull the Gemma model if it's not available"""
    try:
        print("Pulling Gemma model (this may take a while)...")
        subprocess.run(["ollama", "pull", "gemma"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("Ollama command not found. Make sure Ollama is installed correctly.")
        return False


def main():
    """Check Ollama installation and setup"""
    print("Checking Ollama installation...")

    if not check_ollama_installed():
        print("\n❌ Ollama is not installed on this system.")
        print("\nTo install Ollama, visit: https://ollama.com/download")
        print("\nOn macOS, you can use:")
        print("  brew install ollama")
        print("\nOn Linux, you can use:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return False

    print("✅ Ollama is installed.")

    if not check_ollama_running():
        print("\n❌ Ollama service is not running.")
        print("\nStart Ollama with:")
        print("  ollama serve")
        print("\nRun this in a separate terminal window, then try again.")
        return False

    print("✅ Ollama service is running.")

    if not check_gemma_available():
        print("\n❌ Gemma model is not available.")
        print("\nWould you like to pull the Gemma model now? (y/n)")
        choice = input().strip().lower()

        if choice == "y":
            success = pull_gemma_model()
            if not success:
                print("\n❌ Failed to pull Gemma model.")
                print("\nTry manually with:")
                print("  ollama pull gemma")
                return False
        else:
            print("\nYou need to pull the Gemma model before running the script:")
            print("  ollama pull gemma")
            return False

    print("✅ Gemma model is available.")
    print("\nOllama with Gemma is ready to use!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
