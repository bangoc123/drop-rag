import os
import math
import uuid
import tiktoken
import platform
import streamlit as st
import requests
import re


def process_batch(batch_df, model, collection):
    """Encode and save the batch data to Chroma in batches where batch size is specified."""
    try:
        # Encode column data to vectors for this batch
        embeddings = model.encode(batch_df['chunk'].tolist())

        # Collect all metadata in one list (including the newly added '_id' column)
        metadatas = [row.to_dict() for _, row in batch_df.iterrows()]

        # Generate unique ids for the batch
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_df))]

        # Add the batch to Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )


    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'encode'":
            raise RuntimeError("Please set up the language model at section #1 before running the processing.")
        raise RuntimeError(f"Error saving data to Chroma for a batch: {str(e)}")

    
def divide_dataframe(df, batch_size):
    """Divide DataFrame into smaller chunks based on the chunk size."""
    num_batches = math.ceil(len(df) / batch_size)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]



    

# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def clean_collection_name(name):
    # Clean the name based on the required pattern
    # Allow only alphanumeric, underscores, hyphens, and single periods in between
    cleaned_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)   # Step 1: Remove invalid characters
    cleaned_name = re.sub(r'\.{2,}', '.', cleaned_name)    # Step 2: Remove consecutive periods
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned_name)  # Step 3: Remove leading/trailing non-alphanumeric characters

    # Ensure the cleaned name meets length constraints
    return cleaned_name[:63] if 3 <= len(cleaned_name) <= 63 else None



