# Drag and Drop RAG

## Overview

![Application Screenshot](https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/cf0ae020-8b6c-11ef-b095-cf36210a4279-Screen_Shot_2024_10_16_at_10.14.27.png)

[Watch Video Demo](https://www.youtube.com/watch?v=VjVHAfFvnQM&feature=youtu.be)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline. It allows users to upload data files (CSV, JSON, PDF, DOCX), store their content in a **Chroma vector store**, and interact with it through a chatbot powered by **Gemini** or **local open-source models** like those available through **OLLAMA**. The chatbot retrieves relevant information from the uploaded files and uses **LLMs** (Large Language Models) to enhance user queries, returning meaningful responses.

## Features
1. **Upload CSV, JSON, PDF, or DOCX files** – Users can upload files in various formats and choose which columns or sections to index for vector search.
2. **Store and retrieve vector embeddings** using **Chroma** – Automatically store embeddings from uploaded files and retrieve relevant content for queries.
3. **Interactive chatbot** – Use the **Gemini API** or local models to generate contextually enhanced responses.
4. **Customizable LLM options** – Choose between cloud-based **Gemini** or local LLMs, including **OLLAMA** and a range of open-source models.
5. **Flexible chunking options** – Users can apply chunking strategies like **Recursive Token Chunking**, **Agentic Chunking**, or skip chunking altogether.

## Running the Application

### 1. Clone the repository
```bash
git clone https://github.com/bangoc123/drop-rag.git
cd drop-rag
```

### 2. Install the required packages
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Usage Instructions

### 1. **Upload Data**
Upload a CSV, JSON, PDF, or DOCX file. You can select which column(s) to index for vector-based searches. 

### 2. **Embedding and Storage**
The data is stored in the **Chroma vector store**, where vector embeddings are generated using models like `all-MiniLM-L6-v2` (for English) or `keepitreal/vietnamese-sbert` (for Vietnamese).

### 3. **Choosing LLM**
You can choose to use either:
- **Gemini API**: Requires a **Gemini API key** to generate responses. Obtain the key [here](https://aistudio.google.com/app/apikey).
- **Local LLMs via OLLAMA**: Use **OLLAMA** to run models such as `llama`, `gpt-j`, and other open-source models on your local machine.

### OLLAMA Model Options

Here’s a list of models available for use with **OLLAMA**, along with their corresponding identifiers:

| Model Name                            | Model Size        | OLLAMA Identifier       |
|---------------------------------------|-------------------|-------------------------|
| Llama 3.2 (3B - 2.0GB)                | 3B (2.0GB)        | `llama3.2`              |
| Llama 3.2 (1B - 1.3GB)                | 1B (1.3GB)        | `llama3.2:1b`           |
| Llama 3.1 (8B - 4.7GB)                | 8B (4.7GB)        | `llama3.1`              |
| Llama 3.1 (70B - 40GB)                | 70B (40GB)        | `llama3.1:70b`          |
| Llama 3.1 (405B - 231GB)              | 405B (231GB)      | `llama3.1:405b`         |
| Phi 3 Mini (3.8B - 2.3GB)             | 3.8B (2.3GB)      | `phi3`                  |
| Phi 3 Medium (14B - 7.9GB)            | 14B (7.9GB)       | `phi3:medium`           |
| Gemma 2 (2B - 1.6GB)                  | 2B (1.6GB)        | `gemma2:2b`             |
| Gemma 2 (9B - 5.5GB)                  | 9B (5.5GB)        | `gemma2`                |
| Gemma 2 (27B - 16GB)                  | 27B (16GB)        | `gemma2:27b`            |
| Mistral (7B - 4.1GB)                  | 7B (4.1GB)        | `mistral`               |
| Moondream 2 (1.4B - 829MB)            | 1.4B (829MB)      | `moondream`             |
| Neural Chat (7B - 4.1GB)              | 7B (4.1GB)        | `neural-chat`           |
| Starling (7B - 4.1GB)                 | 7B (4.1GB)        | `starling-lm`           |
| Code Llama (7B - 3.8GB)               | 7B (3.8GB)        | `codellama`             |
| Llama 2 Uncensored (7B - 3.8GB)       | 7B (3.8GB)        | `llama2-uncensored`     |
| LLaVA (7B - 4.5GB)                    | 7B (4.5GB)        | `llava`                 |
| Solar (10.7B - 6.1GB)                 | 10.7B (6.1GB)     | `solar`                 |

### 4. **Chatbot Interaction**
After uploading your data and selecting the LLM, start interacting with the chatbot, which will retrieve and augment responses based on the stored data.

### 5. **Chunking Options**
Before querying the chatbot, users can choose from different chunking methods:
- **No Chunking**: Use the full document without dividing it.
- **Recursive Token Chunking**: Split documents into smaller sections based on token count.
- **Semantic Chunking**: Group text by meaning, enhancing retrieval accuracy.
- **Agentic Chunking**: Dynamically manage text chunks using an LLM-based agent.

## Notes
- A **Gemini API key** is required if you opt for the **Gemini** model. 
- For local model inference using **OLLAMA**, ensure **Docker** is installed to run the models locally.

## Troubleshooting

- If queries do not retrieve results, verify that you have selected the correct columns for indexing and that your embeddings are properly stored.
- Ensure the API key is valid (if using Gemini) and that the vector store is initialized before using the chatbot.
