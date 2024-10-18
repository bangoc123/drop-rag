import os
import platform
import requests
import streamlit as st
import subprocess
import json
import os
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the value of OLLAMA_ENDPOINT
ollama_endpoint = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"


# Available models with command and details
OLLAMA_MODEL_OPTIONS = {

    "Llama 3.2 (3B - 2.0GB)": "llama3.2",
    "Llama 3.2 (1B - 1.3GB)": "llama3.2:1b",
    "Llama 3.1 (8B - 4.7GB)": "llama3.1",
    "Llama 3.1 (70B - 40GB)": "llama3.1:70b",
    "Llama 3.1 (405B - 231GB)": "llama3.1:405b",
    "Phi 3 Mini (3.8B - 2.3GB)": "phi3",
    "Phi 3 Medium (14B - 7.9GB)": "phi3:medium",
    "Gemma 2 (2B - 1.6GB)": "gemma2:2b",
    "Gemma 2 (9B - 5.5GB)": "gemma2",
    "Gemma 2 (27B - 16GB)": "gemma2:27b",
    "Mistral (7B - 4.1GB)": "mistral",
    "Moondream 2 (1.4B - 829MB)": "moondream",
    "Neural Chat (7B - 4.1GB)": "neural-chat",
    "Starling (7B - 4.1GB)": "starling-lm",
    "Code Llama (7B - 3.8GB)": "codellama",
    "Llama 2 Uncensored (7B - 3.8GB)": "llama2-uncensored",
    "LLaVA (7B - 4.5GB)": "llava",
    "Solar (10.7B - 6.1GB)": "solar"
}

GGUF_MODEL_OPTIONS ={
    "Llama-3.2-1B-Instruct-GGUF": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
    "SmolLM-1.7B-Instruct-v0.2-GGUF": "hf.co/MaziyarPanahi/SmolLM-1.7B-Instruct-v0.2-GGUF",
}


# Function to install Nvidia Container Toolkit (for Nvidia GPU setup)
def install_nvidia_toolkit():
    st.info("Installing NVIDIA Container Toolkit...")
    os.system("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    os.system("curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
    os.system("sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit")
    os.system("sudo nvidia-ctk runtime configure --runtime=docker")
    os.system("sudo systemctl restart docker")


# Function to check if NVIDIA GPU is available
def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Function to check if AMD GPU is available
def has_amd_gpu():
    try:
        result = subprocess.run(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 'AMD' in result.stdout.decode()
    except FileNotFoundError:
        return False

def remove_running_container(
        container_name,
        position_noti="content"
    ):
    # Check if the container is running
    result = subprocess.run(["docker", "ps", "-q", "--filter", f"name={container_name}"], capture_output=True, text=True)
    if result.stdout.strip():  # Container is running
        os.system(f"docker rm -f {container_name}")
        if position_noti == "content":
            st.success(f"Removed the running container '{container_name}'.")
        else:
            st.sidebar.success(f"Removed the running container '{container_name}'.")

# Function to run the Ollama container based on the hardware type
def run_ollama_container(
        position_noti="content"
    ):
    system = platform.system().lower()
    container_name = "ollama"

    # Remove the container if it's already running
    remove_running_container(
        container_name,
        position_noti=position_noti
    )

    if system == "linux" or system == "darwin":  # macOS or Linux
        if has_nvidia_gpu():
            st.info("NVIDIA GPU detected. Installing NVIDIA Container Toolkit if necessary...")
            install_nvidia_toolkit()  # Ensure NVIDIA toolkit is installed
            # Run Ollama container with NVIDIA GPU
            os.system(f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
            if position_noti == "content":
                st.success("Ollama container running with NVIDIA GPU!")
            else:
                st.sidebar.success("Ollama container running with NVIDIA GPU!")
        elif has_amd_gpu():
            st.info("AMD GPU detected. Starting Ollama with ROCm support...")
            # Run Ollama container with AMD GPU
            os.system(f"docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama:rocm")
            if position_noti == "content":
                st.success("Ollama container running with AMD GPU!")
            else:
                st.sidebar.success("Ollama container running with AMD GPU!")
        else:
            if position_noti == "content":
                st.info("No GPU detected. Starting Ollama with CPU-only support...")
            else:
                st.sidebar.info("No GPU detected. Starting Ollama with CPU-only support...")
            # Run Ollama container with CPU-only
            os.system(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
            
            if position_noti == "content":
                st.success("Ollama container running with CPU-only!")
            else:
                st.sidebar.success("Ollama container running with CPU-only!")

    elif system == "windows":
        if position_noti == "content":
            st.warning("Please download and install Docker Desktop for Windows and run the following command manually:")
        else:
            st.sidebar.warning("Please download and install Docker Desktop for Windows and run the following command manually:")
        st.code(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")


class LocalLlms:
    def __init__(self, model_name, position_noti="content"):
        self.model_name = model_name
        self.base_url = ollama_endpoint
        self.position_noti = position_noti
        self.pull_model()

    def pull_model(self):
        """Pull the specified model from the Ollama server."""
        st.spinner(f"Pulling model {self.model_name}...")
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"model": self.model_name}
        )

        if response.status_code != 200:
            st.error(f"Failed to pull model {self.model_name}: {response.text}")
            raise Exception(f"Model pull failed: {response.text}")

        if self.position_noti == "content":
            st.success(f"Model {self.model_name} pulled successfully.")
        else:
            st.sidebar.success(f"Model {self.model_name} pulled successfully.")

    def chat(self, messages):
        """Send messages to the model and return the assistant's response."""
        try:
            data = {
                "model": self.model_name, 
                "messages": messages,  
                "stream": False,       
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data
            )
            
            # Check if the response is successful
            if response.status_code == 200:
                response_json = response.json()  # Parse JSON response
                
                # Extract the assistant's content
                assistant_message = response_json.get('message', {}).get('content', '')
                
                # Return the assistant's message and other relevant details if needed
                return {
                    "content": assistant_message,
                    "model": response_json.get('model'),
                    "created_at": response_json.get('created_at'),
                    "total_duration": response_json.get('total_duration'),
                    "load_duration": response_json.get('load_duration'),
                    "prompt_eval_count": response_json.get('prompt_eval_count'),
                    "prompt_eval_duration": response_json.get('prompt_eval_duration'),
                    "eval_count": response_json.get('eval_count'),
                    "eval_duration": response_json.get('eval_duration'),
                    "done": response_json.get('done')
                }
            else:
                print(f"Error: Received status code {response.status_code}")
                return None

        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def generate_content(self, prompt):
        data = {
            "model": self.model_name, 
            "prompt": prompt,
            "stream": False,       
        }

        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data
        )

        if response.status_code == 200:
            response_json = response.json()
            return response_json.get("response")
        else:
            return ""

def run_ollama_model(
        model_name="gemma2:2b",
        position_noti="content"
    ):
    # Check if the Ollama server is running
    try:
        response = requests.get(ollama_endpoint)
        if response.status_code != 200:
            if position_noti == "content":
                st.error("Ollama server is not running. Please start the server first.")
            else:
                st.sidebar.error("Ollama server is not running. Please start the server first.")
            return None
    except requests.ConnectionError:
        if position_noti == "content":
            st.error("Ollama server is not reachable. Please check if it's running.")
        else:
            st.sidebar.error("Ollama server is not reachable. Please check if it's running.")
        return None

    # Create and return an instance of LocalLlms
    return LocalLlms(
        model_name,
        position_noti=position_noti
    )
