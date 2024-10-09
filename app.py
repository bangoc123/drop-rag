import streamlit as st
import pandas as pd
import json
import uuid  # For generating unique IDs
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import RecursiveTokenChunker

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



# Step 1: File Upload (CSV or JSON) and Column Detection
uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])

# Initialize a variable for tracking the success of saving the data
st.session_state.data_saved_success = False

if uploaded_file is not None:
    # Determine file type and read accordingly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        json_data = json.load(uploaded_file)
        df = pd.json_normalize(json_data)  # Normalize JSON to a flat DataFrame format
    
    # st.write("Detected Columns:", df.columns.tolist())
    st.dataframe(df)

    doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]

    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = doc_ids

    # Add or replace the '_id' column in the DataFrame
    df['doc_id'] = st.session_state.doc_ids

    st.subheader("Chunking")

    # Step 2: Ask user for the index column (to generate embeddings)
    index_column = st.selectbox("Choose the column to index (for vector search):", df.columns)

    chunkOption = st.radio(
        "Please select one of the options below.",
        ["No Chunking", "RecursiveTokenChunker"],
        captions=[
        "Keep the original document",
        "Recursively chunks text into smaller, meaningful token groups based on specific rules or criteria.",
    ],
    )

    chunker = RecursiveTokenChunker(
        chunk_size=200
    )
    chunk_records = []

    # Iterate over rows in the original DataFrame
    for index, row in df.iterrows():

        # For "No Chunking" option, treat the selected index column as a single "chunk"
        if chunkOption == "No Chunking":
            # Use the selected index_column
            selected_column_value = row[index_column]  # Dynamically use the selected column for chunking
            if type(selected_column_value) == str and len(selected_column_value) > 0:
                # Include all original columns in the chunk record
                chunk_record = {**row.to_dict(), 'chunk': selected_column_value}
                
                # Rearrange the dictionary to ensure 'chunk' and '_id' come first
                chunk_record = {
                    'chunk': chunk_record['chunk'],
                    # '_id': str(uuid.uuid4()),
                    **{k: v for k, v in chunk_record.items() if k not in ['chunk', '_id']}
                }
                chunk_records.append(chunk_record)

        # For "RecursiveTokenChunker" option, split text from the selected index column into smaller chunks
        elif chunkOption == "RecursiveTokenChunker":
            selected_column_value = row[index_column]  # Dynamically use the selected column for chunking
            if type(selected_column_value) == str and len(selected_column_value) > 0:
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


    if st.button("Save Data"):
        try:
            # Encode column data to vectors
            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = st.session_state.model.encode(chunks_df['chunk'].tolist())

            # Collect all metadata in one list (including the newly added '_id' column)
            metadatas = [row.to_dict() for _, row in chunks_df.iterrows()]

            # Insert all records into the Chroma collection in a single call
            chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunks_df))]

            st.session_state.collection.add(
                ids=chunk_ids,               # unique ids
                embeddings=embeddings, # vector representations
                metadatas=metadatas    # metadata for each record
            )

            st.success("Data saved to Chroma vector store successfully!")
            st.session_state.data_saved_success = True  # Mark data as saved successfully


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

# Define a helper function for formatting retrieved data
def get_search_result(query, collection, columns_to_answer):
    query_embeddings = st.session_state.model.encode([query])
    search_results = collection.query(query_embeddings=query_embeddings, n_results=10)  # Fetch top 10 results
    search_result = ""

    metadatas =  search_results['metadatas']

    i = 0
    for meta in metadatas[0]:
        i += 1
        search_result += f"\n{i})"
        for column in columns_to_answer:
            if column in meta:
                search_result += f" {column.capitalize()}: {meta.get(column)}"
                # search_result += "-------------------"

        search_result += "\n"
    return search_result

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
                enhanced_prompt = prompt + "\n\nRetrieved Data:\n" + get_search_result(prompt, st.session_state.collection, columns_to_answer)

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

