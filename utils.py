import math
import uuid
import tiktoken

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
        raise RuntimeError(f"Error saving data to Chroma for a batch: {str(e)}")

    
def divide_dataframe(df, batch_size):
    """Divide DataFrame into smaller chunks based on the chunk size."""
    num_batches = math.ceil(len(df) / batch_size)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


# Define a helper function for formatting retrieved data
def get_search_result(model, query, collection, columns_to_answer):
    query_embeddings = model([query])
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

        search_result += "\n"
    return search_result


# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens
