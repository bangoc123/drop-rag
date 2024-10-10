import streamlit as st
import pandas as pd
import json
import uuid  # For generating unique IDs
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import RecursiveTokenChunker, LLMAgenticChunker, ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, get_search_result
import time
import pdfplumber  # PDF extraction
import io

# Initialize the page
st.title("Drag and Drop RAG")
st.logo("https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/678dadd0-603b-11ef-b0a7-998b84b38d43-ProtonX_logo_horizontally__1_.png")

# Initialize session state for chroma client, collection, and model
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")

# Initialize session state for collection and model
if "collection" not in st.session_state:
    st.session_state.collection = None

if "model" not in st.session_state:
    st.session_state.model = None

# Check if the collection exists, if not, create a new one
if st.session_state.collection is None:
    st.session_state.collection = st.session_state.client.get_or_create_collection("rag_collection")

# Step 1: File Upload (CSV, JSON, or PDF) and Column Detection
uploaded_file = st.file_uploader("Upload CSV, JSON, or PDF file", type=["csv", "json", "pdf"])

# Initialize a variable for tracking the success of saving the data
st.session_state.data_saved_success = False

if uploaded_file is not None:
    # Determine file type and read accordingly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        json_data = json.load(uploaded_file)
        df = pd.json_normalize(json_data)  # Normalize JSON to a flat DataFrame format
    elif uploaded_file.name.endswith(".pdf"):
        # Extract text from PDF
        pdf_text = []
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                pdf_text.append(page.extract_text())

        # Convert PDF text into a DataFrame (assuming one column for simplicity)
        df = pd.DataFrame({"content": pdf_text})
    
    st.dataframe(df)

    doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]

    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = doc_ids

    # Add or replace the '_id' column in the DataFrame
    df['doc_id'] = st.session_state.doc_ids

    st.subheader("Chunking")
    # Step 2: Input Gemini API key (only needed for AgenticChunker)
    gemini_api_key = st.text_input("Enter your Gemini API Key (required for AgenticChunker):", type="password")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        st.success("Gemini API Key saved successfully!")
        st.session_state.gemini_api_key = gemini_api_key
    else:
        st.warning("Please enter the API key for AgenticChunker.")

    # Step 2: Ask user for the index column (to generate embeddings)
    index_column = st.selectbox("Choose the column to index (for vector search):", df.columns)

    # Disable the "AgenticChunker" option if the API key is not provided
    chunk_options = ["No Chunking", "RecursiveTokenChunker", "SemanticChunker"]
    if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
        chunk_options.append("AgenticChunker")
    else:
        st.warning("AgenticChunker will only be available after entering the Gemini API key.")
    
    # Step 4: Chunking options
    chunkOption = st.radio(
        "Please select one of the options below.",
        chunk_options,
        captions=[
            "Keep the original document",
            "Recursively chunks text into smaller, meaningful token groups based on specific rules or criteria.",
            "Chunking with semantic comparison between chunks",
            "Let LLM decide chunking (requires Gemini API)"
        ]
    )
    
    if chunkOption == "SemanticChunker":
        embedding_option = st.selectbox(
            "Choose the embedding method for Semantic Chunker:",
            ["TF-IDF", "Sentence-Transformers"]
        )
    chunk_records = []

    # Iterate over rows in the original DataFrame
    for index, row in df.iterrows():

        # For "No Chunking" option, treat the selected index column as a single "chunk"
        chunker = None
        selected_column_value = row[index_column]
        chunks = []
        if not(type(selected_column_value) == str and len(selected_column_value) > 0):
            continue
        if chunkOption == "No Chunking":
            # Use the selected index_column
            chunks = [selected_column_value]
            
        # For "RecursiveTokenChunker" option, split text from the selected index column into smaller chunks
        elif chunkOption == "RecursiveTokenChunker":
            chunker = RecursiveTokenChunker(
                chunk_size=200
            )
            chunks = chunker.split_text(selected_column_value)
            
        elif chunkOption == "SemanticChunker":
            if embedding_option == "TF-IDF":
                chunker = ProtonxSemanticChunker(embedding_type="tfidf")
            else:
                chunker = ProtonxSemanticChunker(embedding_type="transformers", model="all-MiniLM-L6-v2")
            chunks = chunker.split_text(selected_column_value)
        elif chunkOption == "AgenticChunker":
            chunker = LLMAgenticChunker(organisation="google", model_name="gemini-1.5-pro", api_key=gemini_api_key)
            chunks = chunker.split_text(selected_column_value)
        # For each chunk, add a dictionary with the chunk and original_id to the list
        for chunk in chunks:
            chunk_record = {**row.to_dict(), 'chunk': chunk}
            
            # Rearrange the dictionary to ensure 'chunk' and '_id' come first
            chunk_record = {
                'chunk': chunk_record['chunk'],
                # '_id': str(uuid.uuid4()),
                **{k: v for k, v in chunk_record.items() if k not in ['chunk', '_id']}
            }
            chunk_records.append(chunk_record)

    # Convert the list of dictionaries to a DataFrame
    chunks_df = pd.DataFrame(chunk_records)

    # Display the result
    st.write("Number of chunks:", len(chunks_df))
    st.dataframe(chunks_df)



# Button to save data
# Button to save data
if st.button("Save Data"):
    try:
        # Initialize the model and collection
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
        collection = st.session_state.collection

        # Define the batch size
        batch_size = 256

        # Split the DataFrame into smaller batches
        df_batches = divide_dataframe(chunks_df, batch_size)

        # Check if the dataframe has data, otherwise show a warning and skip the processing
        if not df_batches:
            st.warning("No data available to process.")
        else:
            num_batches = len(df_batches)

            # Initialize progress bar
            progress_text = "Saving data to Chroma. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            # Process each batch
            for i, batch_df in enumerate(df_batches):
                if batch_df.empty:
                    continue  # Skip empty batches (just in case)

                process_batch(batch_df, st.session_state.model, collection)

                # Update progress dynamically for each batch
                progress_percentage = int(((i + 1) / num_batches) * 100)
                my_bar.progress(progress_percentage, text=f"Processing batch {i + 1}/{num_batches}")

                time.sleep(0.1)  # Optional sleep to simulate processing time

            # Empty the progress bar once completed
            my_bar.empty()

            st.success("Data saved to Chroma vector store successfully!")
            st.session_state.data_saved_success = True

    except Exception as e:
        st.error(f"Error saving data to Chroma: {str(e)}")


# Show blue tick if data has been saved successfully
header_i = 1
header_text = "{}. Setup data ✅".format(header_i) if st.session_state.data_saved_success else "{}. Setup data".format(header_i)
st.header(header_text)



if st.session_state.data_saved_success:
    st.markdown("✅ **Data Saved Successfully!**")



# Step 3: Define which columns LLMs should answer from
if uploaded_file:
    columns_to_answer = st.multiselect(
        "Select one or more columns LLMs should answer from (multiple selections allowed):", 
        df.columns
    )

# Step 2: Setup LLMs (Gemini Only)
header_i += 1
header_text_llm = "{}. Setup LLMs ✅".format(header_i) if 'gemini_model' in st.session_state else "{}. Setup LLMs".format(header_i)
st.header(header_text_llm)

# Initialize a variable for tracking if the API key was entered successfully
api_key_success = False

# Input Gemini API key
gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    st.success("Gemini API Key saved successfully!")
    st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    api_key_success = True

# Show blue tick if API key was entered successfully
if api_key_success:
    st.markdown("✅ **API Key Saved Successfully!**")


# Step 3: Interactive Chatbot
header_i += 1
st.header("{}. Interactive Chatbot".format(header_i))

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# URL of the Flask API

# Display the chat history using chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    # Prepare the payload for the request

    with st.chat_message("assistant"):
        if st.session_state.collection is not None:
            # Step 1: Retrieve relevant data using query and vector search
            query_embeddings = st.session_state.model.encode([prompt])

            # Step 2: Combine retrieved data to enhance the prompt based on selected columns
            if columns_to_answer:
                retrieved_data = get_search_result(st.session_state.model, prompt, st.session_state.collection, columns_to_answer)
                
                enhanced_prompt = """You are a good salesperson. The prompt of the customer is: "{}". Answer it based on the following retrieved data: \n{}""".format(prompt, retrieved_data)

                # Step 3: Feed enhanced prompt to Gemini LLM for completion
                response = st.session_state.gemini_model.generate_content(enhanced_prompt)

                content = response.candidates[0].content.parts[0].text

                # Display the extracted content in the Streamlit app
                st.markdown(content)

                # Update chat history
                st.session_state.chat_history.append({"role": "assistant", "content": content})
            else:
                st.warning("Please select columns for the chatbot to answer from.")
        else:
            st.error("No collection found. Please upload data and save it first.")

