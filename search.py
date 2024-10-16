import numpy as np
# Define a helper function for formatting retrieved data
def vector_search(model, query, collection, columns_to_answer, number_docs_retrieval ):
    query_embeddings = model.encode([query])
    search_results = collection.query(
        query_embeddings=query_embeddings, 
        n_results=number_docs_retrieval)  # Fetch top 10 results
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
    return metadatas, search_result


def keywords_search(query, collection, columns_to_answer, number_docs_retrieval):
    search_results = collection.query(
        query_texts=[query],
        n_results=number_docs_retrieval)  # Fetch top 10 results
    search_result = ""
    metadatas = search_results['metadatas']

    i = 0
    for meta in metadatas[0]:
        i += 1
        search_result += f"\n{i})"
        for column in columns_to_answer:
            if column in meta:
                search_result += f" {column.capitalize()}: {meta.get(column)}"

        search_result += "\n"
    return metadatas, search_result

def generate_hypothetical_documents(model, query, num_samples=5):
    """
    Generate multiple hypothetical documents using the Gemini model.

    Parameters:
        model (Gemini): The Gemini model to use for generation.
        query (str): The original search query.
        num_samples (int): Number of hypothetical documents to generate.

    Returns:
        list: A list of generated hypothetical documents.
    """
    hypothetical_docs = []
    for _ in range(num_samples):
        enhanced_prompt = f"Write a paragraph that answers the question: {query}"
        # Use the Gemini model stored in session state to generate the document
        response = model.generate_content(enhanced_prompt)
        hypothetical_docs.append(response)
    
    return hypothetical_docs

def encode_hypothetical_documents(documents, encoder_model):
    """
    Encode multiple hypothetical documents into embeddings using a dense retriever.

    Parameters:
        documents (list): List of hypothetical documents.
        encoder_model (SentenceTransformer): The preloaded SentenceTransformer model.

    Returns:
        np.ndarray: An aggregated embedding vector representing the query.
    """
    # Encode each document into an embedding
    embeddings = [encoder_model.encode([doc])[0] for doc in documents]
    # Average the embeddings to get a single query representation
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def hyde_search(llm_model, encoder_model, query, collection, columns_to_answer, number_docs_retrieval, num_samples=5):
    """
    Search the collection using the HYDE algorithm.
    """
    hypothetical_documents = generate_hypothetical_documents(llm_model, query, num_samples)

    print("hypothetical_documents", hypothetical_documents)
    aggregated_embedding = encode_hypothetical_documents(hypothetical_documents, encoder_model)

   
    search_results = collection.query(
        query_embeddings=aggregated_embedding, 
        n_results=number_docs_retrieval)  # Fetch top 10 results
    
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
    return metadatas, search_result