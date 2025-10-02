import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
import uuid
import numpy as np


class ChromaVectorDB:
    """
    A ChromaDB-based vector database class with persistent storage.
    
    This class provides methods to:
    - Initialize a persistent ChromaDB collection
    - Add documents with embeddings and metadata
    - Query the collection for similar documents
    - Get collection metadata and statistics
    """
    
    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize the ChromaDB vector database with persistent storage.
        
        Args:
            collection_name (str): Name of the collection to create/use
            persist_directory (str): Path where ChromaDB will store data persistently
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Vector database collection: {collection_name}"}
        )
        
        print(f"ChromaDB initialized with collection '{collection_name}' at '{persist_directory}'")
    
    @classmethod
    def create(cls, collection_name: str, persist_directory: str) -> 'ChromaVectorDB':
        """
        Factory method to create a ChromaVectorDB instance.
        
        Args:
            collection_name (str): Name of the collection to create/use
            persist_directory (str): Path where ChromaDB will store data persistently
            
        Returns:
            ChromaVectorDB: A new instance of ChromaVectorDB
            
        Example:
            >>> db = ChromaVectorDB.create("my_documents", "./vector_storage")
            >>> db = ChromaVectorDB.create("research_papers", "/path/to/storage")
        """
        return cls(collection_name=collection_name, persist_directory=persist_directory)
    
    def add_documents(
        self,
        documents: Union[str, List[str]],
        embeddings: Union[List[float], List[List[float]]],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Add one or more documents to the collection.
        
        Args:
            documents (str or List[str]): Document text(s) to add
            embeddings (List[float] or List[List[float]]): Embedding vector(s) for the document(s)
            metadatas (Dict or List[Dict], optional): Metadata for the document(s)
            ids (str or List[str], optional): Custom IDs for the documents. If None, UUIDs will be generated
            
        Returns:
            List[str]: List of document IDs that were added
        """
        # Normalize inputs to lists
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(embeddings[0], (int, float)):  # Single embedding
            embeddings = [embeddings]
        if metadatas is None:
            metadatas = [{}] * len(documents)
        elif isinstance(metadatas, dict):
            metadatas = [metadatas]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif isinstance(ids, str):
            ids = [ids]
        
        # Validate input lengths
        if not (len(documents) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError("All input lists must have the same length")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} document(s) to collection '{self.collection_name}'")
        return ids
    
    def query(
        self,
        query_embeddings: Union[List[float], List[List[float]]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        Args:
            query_embeddings (List[float] or List[List[float]]): Query embedding vector(s)
            n_results (int): Number of results to return per query
            where (Dict, optional): Metadata filter conditions
            where_document (Dict, optional): Document content filter conditions
            include (List[str], optional): What to include in results. 
                                         Options: ["documents", "embeddings", "metadatas", "distances"]
                                         Default: ["documents", "metadatas", "distances"]
            
        Returns:
            Dict[str, Any]: Query results containing documents, metadata, distances, etc.
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        # Normalize query_embeddings to list of lists
        if isinstance(query_embeddings[0], (int, float)):  # Single query
            query_embeddings = [query_embeddings]
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        return results
    
    def query_by_text(
        self,
        query_texts: Union[str, List[str]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection using text (ChromaDB will handle embedding generation).
        
        Args:
            query_texts (str or List[str]): Query text(s)
            n_results (int): Number of results to return per query
            where (Dict, optional): Metadata filter conditions
            where_document (Dict, optional): Document content filter conditions
            include (List[str], optional): What to include in results
            
        Returns:
            Dict[str, Any]: Query results containing documents, metadata, distances, etc.
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata about the ChromaDB collection.
        
        Returns:
            Dict[str, Any]: Collection information including name, path, count, etc.
        """
        # Get collection count
        count = self.collection.count()
        
        # Get collection metadata
        collection_metadata = self.collection.metadata or {}
        
        # Get a sample of documents to understand structure (if any exist)
        sample_data = None
        if count > 0:
            try:
                sample_data = self.collection.peek(limit=1)
            except Exception as e:
                sample_data = f"Error retrieving sample: {e}"
        
        info = {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "document_count": count,
            "collection_metadata": collection_metadata,
            "sample_data": sample_data,
            "client_settings": {
                "path": str(self.persist_directory),
                "anonymized_telemetry": False
            }
        }
        
        return info
    
    def print_collection_info(self):
        """
        Print formatted collection information.
        """
        info = self.get_collection_info()
        
        print(f"\n{'='*60}")
        print(f"CHROMADB COLLECTION INFORMATION")
        print(f"{'='*60}")
        print(f"Collection Name: {info['collection_name']}")
        print(f"Storage Path: {info['persist_directory']}")
        print(f"Document Count: {info['document_count']:,}")
        print(f"Collection Metadata: {info['collection_metadata']}")
        
        if info['sample_data'] and isinstance(info['sample_data'], dict):
            print(f"\nSample Data Structure:")
            for key, value in info['sample_data'].items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {type(value[0]).__name__} (length: {len(value)})")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        print(f"{'='*60}\n")
    
    def delete_collection(self):
        """
        Delete the entire collection and all its data.
        
        Warning: This operation is irreversible!
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' has been deleted.")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        """
        Reset the collection by deleting and recreating it.
        
        Warning: This will remove all documents from the collection!
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Vector database collection: {self.collection_name}"}
            )
            print(f"Collection '{self.collection_name}' has been reset.")
        except Exception as e:
            print(f"Error resetting collection: {e}")
    
    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve specific documents by their IDs.
        
        Args:
            ids (List[str]): List of document IDs to retrieve
            
        Returns:
            Dict[str, Any]: Documents and their associated data
        """
        return self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update existing documents in the collection.
        
        Args:
            ids (List[str]): IDs of documents to update
            documents (List[str], optional): New document texts
            embeddings (List[List[float]], optional): New embeddings
            metadatas (List[Dict], optional): New metadata
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Updated {len(ids)} document(s) in collection '{self.collection_name}'")
    
    def delete_documents(self, ids: List[str]):
        """
        Delete specific documents from the collection.
        
        Args:
            ids (List[str]): IDs of documents to delete
        """
        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} document(s) from collection '{self.collection_name}'")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the vector database
    db = ChromaVectorDB.create(
        "test_collection",  "./chroma_db"
    )
    
    # Print initial collection info
    db.print_collection_info()
    
    # Example documents and embeddings (you would typically get embeddings from your EmbeddingModel)
    sample_documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Mock embeddings (in practice, you'd use your EmbeddingModel to generate these)
    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 5-dimensional for example
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    
    sample_metadata = [
        {"topic": "AI", "source": "textbook", "chapter": 1},
        {"topic": "Deep Learning", "source": "research paper", "year": 2023},
        {"topic": "NLP", "source": "tutorial", "difficulty": "beginner"}
    ]
    
    # Add documents
    doc_ids = db.add_documents(
        documents=sample_documents,
        embeddings=sample_embeddings,
        metadatas=sample_metadata
    )
    
    print(f"Added documents with IDs: {doc_ids}")
    
    # Print updated collection info
    db.print_collection_info()
    
    # Example query (using mock embedding)
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
    results = db.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    
    print("Query Results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"{i+1}. Document: {doc}")
        print(f"   Metadata: {metadata}")
        print(f"   Distance: {distance:.4f}\n")
