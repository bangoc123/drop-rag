# Drag and Drop RAG

## Overview

![Application Screenshot](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/cf0ae020-8b6c-11ef-b095-cf36210a4279-Screen_Shot_2024_10_16_at_10.14.27.png)

[Watch Video Demo](https://www.youtube.com/watch?v=VjVHAfFvnQM&feature=youtu.be)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline, enabling users to upload various data files (CSV, JSON, PDF, DOCX), store their content in a **Chroma vector store**, and interact with it via a chatbot. The chatbot is powered by **Gemini**, **OpenAI**, or local models accessible through **OLLAMA**, retrieving relevant information and using **Large Language Models (LLMs)** to enhance responses based on user queries. Additionally, the app provides an **experimental Graph RAG** feature for visualizing data connections.

## Features
1. **Flexible File Upload** – Supports uploading CSV, JSON, PDF, and DOCX files, allowing users to choose which columns or sections to index.
2. **Chroma-based Storage and Retrieval** – Uses **Chroma** to store vector embeddings and perform efficient vector-based searches.
3. **Interactive Chatbot** – Chatbot interaction is enhanced by **Gemini**, **OpenAI**, or local LLMs to generate context-aware responses.
4. **Customizable LLM Choices** – Offers options between cloud-based **Gemini** and **OpenAI**, or local LLMs like **OLLAMA**, with support for various open-source models.
5. **Dynamic Chunking Options** – Provides multiple chunking strategies: **Recursive Token Chunking**, **Agentic Chunking**, **Semantic Chunking**, or no chunking.
6. **Graph RAG Visualization** – Experimental support for visualizing data relationships and connections using **Graph RAG**.

## Running the Application

### 1. Clone the Repository
```bash
git clone https://github.com/bangoc123/drop-rag.git
cd drop-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Usage Instructions

### Step 1: Upload Data
Upload a CSV, JSON, PDF, or DOCX file. You can specify which columns to index for vector-based search.

### Step 2: Embedding and Storage
The data is stored in **Chroma**, and embeddings are generated using models like `all-MiniLM-L6-v2` (for English) or `keepitreal/vietnamese-sbert` (for Vietnamese).

### Step 3: Choose LLM
Select from:
- **Gemini API** (requires a [Gemini API key](https://aistudio.google.com/app/apikey))
- **OpenAI API** (requires an [OpenAI API key](https://platform.openai.com/account/api-keys))
- **Local LLMs via OLLAMA**, supporting models like `llama`, `gpt-j`, and more.

### Step 4: Configure Chunking
Select a chunking method to organize the content:
- **No Chunking**: Use the entire document.
- **Recursive Token Chunking**: Divide text based on token count.
- **Semantic Chunking**: Group text semantically.
- **Agentic Chunking**: Use an LLM to dynamically manage text chunks (requires Gemini API).

### Step 5: Interact with the Chatbot
Start chatting with the bot, which will enhance responses using the retrieved content.

### Step 6: Graph RAG (Experimental)
Use **Graph RAG** to visualize relationships and connections within the uploaded data:
1. Ensure an online LLM is configured (Gemini or OpenAI).
2. Click the "Extract Graph" button to generate and display the graph.

## Supported Models for OLLAMA

Here is a list of models supported by **OLLAMA**:

| Model Name                           | Size          | Identifier               |
|--------------------------------------|---------------|--------------------------|
| Llama 3.2 (3B)                       | 3B (2.0GB)    | `llama3.2`               |
| Phi 3 Medium (14B)                   | 14B (7.9GB)   | `phi3:medium`            |
| Code Llama (7B)                      | 7B (3.8GB)    | `codellama`              |
| Mistral (7B)                         | 7B (4.1GB)    | `mistral`                |
| ...                                  | ...           | ...                      |

### Using GGUF Models
High-performance GGUF models are supported. Refer to [Hugging Face](https://huggingface.co/models?library=gguf&sort=trending) for available models.

## Advanced Options
### Graph RAG
Experimental support for **Graph RAG** allows the visualization of data connections. Requires an online LLM (Gemini or OpenAI).

### Search Methods
Choose from:
- **Vector Search**: Based on vector similarity.
- **Hyde Search**: Uses a generative model for improved search accuracy.

## Troubleshooting
- **No Results?** Ensure you've indexed the correct columns and stored embeddings.
- **API Issues?** Verify that your API key is valid (if using Gemini or OpenAI) and that your vector store is initialized.

## Notes
- **Gemini and OpenAI API Keys**: Required for cloud-based LLMs. Obtain keys from their respective platforms.
- **Local Models**: Requires **Docker** for local model inference.

## Exporting Chatbot Configuration
The app allows exporting configuration data for local LLMs to JSON for easy deployment.

## Clearing Session State
Users can clear session state via a sidebar button to reset settings.

