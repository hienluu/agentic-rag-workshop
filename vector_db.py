import argparse
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
import uuid
import numpy as np
from dotenv import load_dotenv
from embedding_model import EmbeddingModel


class ChromaVectorDB:
    """
    A ChromaDB-based vector database class with persistent storage.
    
    This class provides methods to:
    - Initialize a persistent ChromaDB collection
    - Add documents with embeddings and metadata
    - Query the collection for similar documents
    - Get collection metadata and statistics
    """
    
    def __init__(self, collection_name: str, persist_directory: str, desc : Optional[str] = None):
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
        if not desc:
            desc = f"Vector database collection: {collection_name}"

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": desc}
        )
        
        print(f"ChromaDB initialized with collection '{collection_name}' at '{persist_directory}'")
    
    @classmethod
    def create(cls, collection_name: str, persist_directory: str, desc: Optional[str] = None) -> 'ChromaVectorDB':
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
        
        #print(f"Added {len(documents)} document(s) to collection '{self.collection_name}'")
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


def run_interactive_console(collection_name: str, persist_directory: str, n_results: int = 5):
    """
    Run an interactive console for vector database operations.
    
    Args:
        collection_name (str): Name of the collection
        persist_directory (str): Path to the persist directory
        n_results (int): Default number of results to return for queries
    """
    print(f"========== Vector DB Console: {collection_name} ==========")
    load_dotenv()
    vector_db = ChromaVectorDB.create(collection_name, persist_directory)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    embedding_model = None
    
    print("\nWelcome to the Vector Database Interactive Console!")
    print("=" * 60)
    print("Available commands:")
    print("  /i              - Display collection information")
    print("  /d              - Delete the collection")
    print("  /q <query>      - Query the vector database")
    print("  /h or /help     - Show this help message")
    print("  quit or exit    - Exit the console")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("VectorDB> ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Check for quit/exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting Vector Database Console. Goodbye!")
                break
            
            # Check if command starts with /
            if not user_input.startswith('/'):
                print("Error: Commands must start with '/'. Type '/h' for help.")
                continue
            
            # Parse the command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            # Handle info command
            if command == '/i':
                vector_db.print_collection_info()
            
            # Handle delete command
            elif command == '/d':
                confirmation = input("Are you sure you want to delete the collection? (yes/no): ").strip().lower()
                if confirmation in ['yes', 'y']:
                    print(f"Deleting collection: {collection_name} at {persist_directory}")
                    vector_db.delete_collection()
                    print("Collection deleted. Exiting console...")
                    break
                else:
                    print("Delete operation cancelled.")
            
            # Handle query command
            elif command == '/n':
                if len(parts) < 2:
                    print("Error: Please provide a count for set the number of results")
                    continue

                count = parts[1]
                n_results = int(count)
                print(f"set result count to {n_results}\n")
            # Handle query command
            elif command == '/q':
                if len(parts) < 2:
                    print("Error: Please provide a query string. Usage: /q <query>")
                    continue
                
                query_text = parts[1]
                print(f"Querying for: '{query_text}' (returning top {n_results} results)")
                
                # Initialize embedding model if not already done
                if embedding_model is None:
                    print(f"Loading embedding model: {embedding_model_name}")
                    embedding_model = EmbeddingModel.from_name(embedding_model_name)
                
                # Generate query embeddings
                query_embeddings = embedding_model.generate_embeddings(query_text)
                
                # Perform query
                results = vector_db.query(query_embeddings, n_results=n_results)
                
                # Display results
                print("\n" + "=" * 60)
                print("Query Results:")
                print("=" * 60)
                
                if results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    )):
                        print(f"\n{i+1}. Document:")
                        print(f"   {doc}")
                        print(f"   Metadata: {metadata}")
                        print(f"   Distance: {distance:.4f}")
                    print("\n" + "=" * 60 + "\n")
                else:
                    print("No results found.\n")
            
            # Handle help command
            elif command in ['/h', '/help']:
                print("\nAvailable commands:")
                print("  /i              - Display collection information")
                print("  /d              - Delete the collection")
                print("  /n              - Set result count (default is 5)")
                print("  /q <query>      - Query the vector database")
                print("  /h or /help     - Show this help message")
                print("  quit or exit    - Exit the console")
                print()
            
            # Unknown command
            else:
                print(f"Error: Unknown command '{command}'. Type '/h' for help.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with commands.")
        except Exception as e:
            print(f"Error: {e}")


def main(collection_name: str, persist_directory: str, info: bool, delete: bool, query: Optional[str] = None, n_results: int = 5):
    print(f"========== main: {collection_name}, {persist_directory} ==========")
    load_dotenv()
    vector_db = ChromaVectorDB.create(collection_name, persist_directory)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

    if info:
        vector_db.print_collection_info()
    elif delete:
        print(f"Deleting collection: {collection_name} at {persist_directory}")
        vector_db.delete_collection()
    elif query:
        print(f"query: {query} and looking for {n_results} results")
        print(f"embedding_model_name: {embedding_model_name}")
        embedding_model = EmbeddingModel.from_name(embedding_model_name)
        query_embeddings = embedding_model.generate_embeddings(query)

                
        results = vector_db.query(query_embeddings, n_results=n_results)
        print("Query Results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"{i+1}. Document: {doc}")
            print(f"   Metadata: {metadata}")
            print(f"   Distance: {distance:.4f}\n")
    else:
        print("No query or info provided.")

# Example usage and testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vector database operations. Use --interactive for console mode."
    )
    parser.add_argument("--collection_name", "-c", help="Vector database collection name to use", required=True)
    parser.add_argument("--persist_directory", "-p", help="Vector database persist directory to use", required=True)
    parser.add_argument("--interactive", "-I", help="Start interactive console mode", action="store_true")
    parser.add_argument("--info", "-i", help="Display collection information", action="store_true")
    parser.add_argument("--delete", "-d", help="Delete collection information", action="store_true")
    parser.add_argument("--query", "-q", help="Query to search for", required=False)
    parser.add_argument("--n_results", "-n", help="Number of results to return", required=False, default=5, type=int)
    args = parser.parse_args()

    if not args.collection_name and not args.persist_directory:
        print("Please provide --collection_name and --persist_directory arguments.")
        raise ValueError("Please provide --collection_name and --persist_directory arguments.")
    
    if not os.path.isdir(args.persist_directory):
        print(f"Directory {args.persist_directory} does not exist.")
        raise ValueError(f"Directory {args.persist_directory} does not exist.")
    
    # Check if interactive mode is requested
    if args.interactive:
        run_interactive_console(args.collection_name, args.persist_directory, args.n_results)
    elif args.info or args.query or args.delete:
        # Use the traditional single-command mode
        main(args.collection_name, args.persist_directory, args.info, args.delete, args.query, args.n_results)
    else:
        print("Please provide --interactive flag for console mode, or --info, --query, or --delete for single operations.")
        print("Use --help for more information.")
        raise ValueError("No operation specified.")
    
