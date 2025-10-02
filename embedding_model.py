from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Union
import torch


class EmbeddingModel:
    """
    A wrapper class for sentence-transformers models that provides embedding generation,
    similarity calculation, and model information display capabilities.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the model to download from HuggingFace
            device (str): Device to run the model on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model loaded successfully!")
    
    @classmethod
    def from_name(cls, model_name: str, device: str = None) -> 'EmbeddingModel':
        """
        Factory method to create an EmbeddingModel instance from a model name.
        
        Args:
            model_name (str): Name of the model to download from HuggingFace
            device (str): Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            
        Returns:
            EmbeddingModel: A new instance of EmbeddingModel
            
        Example:
            >>> model = EmbeddingModel.from_name("all-MiniLM-L6-v2")
            >>> model = EmbeddingModel.from_name("sentence-transformers/all-mpnet-base-v2", device="cuda")
        """
        return cls(model_name=model_name, device=device)
    
    def generate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for given text(s).
        
        Args:
            text (str or List[str]): Input text or list of texts to embed
            
        Returns:
            np.ndarray: Embedding vector(s) as numpy array
        """
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarities(self, query: str, texts: List[str]) -> List[float]:
        """
        Calculate cosine similarities between a query string and a list of texts.
        
        Args:
            query (str): Query string to compare against
            texts (List[str]): List of texts to compare with the query
            
        Returns:
            List[float]: List of similarity scores (0-1, where 1 is most similar)
        """
        # Generate embeddings
        query_embedding = self.generate_embeddings(query)
        text_embeddings = self.generate_embeddings(texts)
        
        # Ensure query_embedding is 2D for cosine_similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        return similarities.tolist()
    
    def print_model_info(self):
        """
        Print information about the embedding model including parameters,
        max sequence length, and embedding dimension.
        """
        print(f"\n{'='*50}")
        print(f"EMBEDDING MODEL INFORMATION")
        print(f"{'='*50}")
        print(f"Model Name: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Get model configuration
        try:
            # Get the underlying transformer model
            transformer_model = self.model[0].auto_model
            config = transformer_model.config
            
            # Calculate number of parameters
            total_params = sum(p.numel() for p in transformer_model.parameters())
            trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
            
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            
            # Get max sequence length
            max_seq_length = getattr(config, 'max_position_embeddings', 
                                   getattr(config, 'max_seq_length', 'Unknown'))
            print(f"Max Sequence Length: {max_seq_length}")
            
            # Get embedding dimension
            embedding_dim = config.hidden_size
            print(f"Embedding Dimension: {embedding_dim}")
            
            # Additional model info
            if hasattr(config, 'model_type'):
                print(f"Model Type: {config.model_type}")
            if hasattr(config, 'num_hidden_layers'):
                print(f"Number of Layers: {config.num_hidden_layers}")
            if hasattr(config, 'num_attention_heads'):
                print(f"Attention Heads: {config.num_attention_heads}")
                
        except Exception as e:
            print(f"Could not retrieve detailed model information: {e}")
            
        # Get actual max sequence length from sentence-transformers
        try:
            max_seq_len = self.model.get_max_seq_length()
            print(f"Sentence-Transformers Max Length: {max_seq_len}")
        except:
            pass
            
        print(f"{'='*50}\n")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the model
    embedding_model = EmbeddingModel.from_name("all-MiniLM-L6-v2")
    
    # Print model information
    embedding_model.print_model_info()
    
    # Test embedding generation
    test_text = "This is a sample text for embedding generation."
    embedding = embedding_model.generate_embeddings(test_text)
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test similarity calculation
    query = "machine learning algorithms"
    texts = [
        "Deep learning neural networks",
        "Natural language processing techniques", 
        "Computer vision applications",
        "Data science methodologies",
        "Software engineering practices"
    ]
    
    similarities = embedding_model.calculate_similarities(query, texts)
    print(f"\nSimilarity scores for query: '{query}'")
    for i, (text, score) in enumerate(zip(texts, similarities)):
        print(f"{i+1}. {text}: {score:.4f}")
