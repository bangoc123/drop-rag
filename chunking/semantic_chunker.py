from .base_chunker import BaseChunker
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


nltk.download("punkt")



class ProtonxSemanticChunker(BaseChunker):
    def __init__(self, threshold=0.3, embedding_type="tfidf", model="all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.embedding_type = embedding_type
        self.model = model
    
    def embed_function(self, sentences):
        """
        Embeds sentences using the specified embedding method.
        Supports 'tfidf' and 'transformers' embeddings.
        """
        if self.embedding_type == "tfidf":
            vectorizer = TfidfVectorizer().fit_transform(sentences)
            return vectorizer.toarray()
        elif self.embedding_type == "transformers":
            self.model = SentenceTransformer(self.model)
            return self.model.encode(sentences)
        
        else:
            raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'transformers'.")
    
        
    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)  # Extract sentences

        # Vectorize the sentences for similarity checking
        vectors = self.embed_function(sentences)

        # Calculate pairwise cosine similarity between sentences
        similarities = cosine_similarity(vectors)

        # Initialize chunks with the first sentence
        chunks = [[sentences[0]]]

        # Group sentences into chunks based on similarity threshold
        for i in range(1, len(sentences)):
            sim_score = similarities[i-1, i]

            if sim_score >= self.threshold:
                # If the similarity is above the threshold, add to the current chunk
                chunks[-1].append(sentences[i])
            else:
                # Start a new chunk
                chunks.append([sentences[i]])

        # Join the sentences in each chunk to form coherent paragraphs
        return [' '.join(chunk) for chunk in chunks]
        