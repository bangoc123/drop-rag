import streamlit as st
import pandas as pd
import json
import uuid  # For generating unique IDs
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown

# Initialize the page
st.title("RAG Pipeline Setup")
st.logo("https://storage.googleapis.com/mle-courses-prod/users/61b6fa1ba83a7e37c8309756/private-files/678dadd0-603b-11ef-b0a7-998b84b38d43-ProtonX_logo_horizontally__1_.png")

# Initialize session state for chroma client, collection, and model
if "client" not in st.session_state:
    st.session_state.client = chroma = chromadb.Client(tenant="default_tenant")

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

    # Step 2: Ask user for the index column (to generate embeddings)
    index_column = st.selectbox("Choose the column to index (for vector search):", df.columns)

    if st.button("Save Data"):
        try:
            # Encode column data to vectors
            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = st.session_state.model.encode(df[index_column].tolist())

            # Auto-generate IDs for each record
            ids = [str(uuid.uuid4()) for _ in range(len(df))]

            # Insert records into the Chroma collection
            for idx, row in df.iterrows():
                st.session_state.collection.add(
                    ids=[str(ids[idx])],  # unique id
                    embeddings=[embeddings[idx]],  # vector representation
                    metadatas=[row.to_dict()]  # metadata for each record
                )

            st.success("Data saved to Chroma vector store successfully!")
            st.session_state.data_saved_success = True  # Mark data as saved successfully

        except Exception as e:
            st.error(f"Error saving data to Chroma: {str(e)}")

# Show blue tick if data has been saved successfully
header_text = "1) Setup data ✅" if st.session_state.data_saved_success else "1) Setup data"
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
header_text_llm = "2) Setup LLMs ✅" if 'gemini_model' in st.session_state else "2) Setup LLMs"
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
    search_results = collection.query(query_embeddings=query_embeddings, n_results=2)  # Fetch top 10 results
    search_result = ""

    print('****search_results****', search_results)

    metadatas =  search_results['metadatas']

    print('type---', type(metadatas))

    i = 0
    for meta in metadatas[0]:
        i += 1
        search_result += f"\n{i})"
        for column in columns_to_answer:
            if column in meta:
                search_result += f" {column.capitalize()}: {meta.get(column)}"
        
        if 'price' in columns_to_answer and not meta.get('price'):
            search_result += f", Price: Contact for more information"
        search_result += "\n"

    return search_result

# Step 3: Interactive Chatbot
st.header("3) Interactive Chatbot")

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

