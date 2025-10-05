import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, asdict
# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class RagLLM:
    """
    A wrapper class for LLM interactions in the context of RAG (Retrieval-Augmented Generation) applications.
    
    This class provides methods for:
    - Generating responses with retrieved context
    - Simple text completion
    - Streaming responses
    - Managing conversation history
    """
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        Initialize the RAG LLM client.
        
        Args:
            base_url (str): Base URL for the LLM API endpoint (e.g., "https://api.openai.com/v1")
            api_key (str): API key for authentication
            model_name (str): Name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        print(f"RAG LLM initialized with model '{model_name}' at '{base_url}'")
    
    @classmethod
    def create(cls, base_url: str, api_key: str, model_name: str) -> 'RagLLM':
        """
        Factory method to create a RagLLM instance.
        
        Args:
            base_url (str): Base URL for the LLM API endpoint
            api_key (str): API key for authentication
            model_name (str): Name of the model to use
            
        Returns:
            RagLLM: A new instance of RagLLM
            
        Example:
            >>> llm = RagLLM.create(
            ...     base_url="https://api.openai.com/v1",
            ...     api_key="sk-...",
            ...     model_name="gpt-4"
            ... )
        """
        return cls(base_url=base_url, api_key=api_key, model_name=model_name)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM for a given prompt.
        
        Args:
            prompt (str): The user prompt/query
            system_prompt (str, optional): System prompt to set behavior/context
            temperature (float): Sampling temperature (0.0-2.0). Higher = more random
            max_tokens (int, optional): Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Generated response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        context_separator: str = "\n\n---\n\n",
        **kwargs
    ) -> str:
        """
        Generate a response using retrieved context documents (RAG pattern).
        
        Args:
            query (str): User's query/question
            context_documents (List[str]): List of relevant documents retrieved from vector DB
            system_prompt (str, optional): System prompt. If None, uses default RAG prompt
            temperature (float): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate
            context_separator (str): Separator between context documents
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Generated response based on the provided context
        """
        # Format context documents
        formatted_context = context_separator.join(context_documents)
        
        # Default RAG system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on the "
                "provided context. If the answer cannot be found in the context, say so. "
                "Always cite the relevant parts of the context when answering."
            )
        
        # Create RAG prompt with context
        rag_prompt = f"""Context information is below:
{formatted_context}

Question: {query}

Answer:"""
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response given a conversation history.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content' keys
                Example: [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
            temperature (float): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Generated response
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.
        
        Args:
            prompt (str): The user prompt/query
            system_prompt (str, optional): System prompt to set behavior/context
            temperature (float): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            str: Generated text chunks
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured LLM.
        
        Returns:
            Dict[str, Any]: Model configuration information
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_configured": bool(self.api_key)
        }
    
    def print_model_info(self):
        """
        Print formatted information about the LLM configuration.
        """
        info = self.get_model_info()
        
        print(f"\n{'='*50}")
        print(f"RAG LLM INFORMATION")
        print(f"{'='*50}")
        print(f"Model Name: {info['model_name']}")
        print(f"Base URL: {info['base_url']}")
        print(f"API Key Configured: {info['api_configured']}")
        print(f"{'='*50}\n")

@dataclass
class LLMProviderInfo:
    base_url: str
    api_key: str
    model_name: str


def get_llm_provider(provider: str) -> LLMProviderInfo:
    if provider == "gemini":
        return LLMProviderInfo(
            base_url=os.getenv("GEMINI_API_URL"),
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_MODEL_NAME")
        )
    elif provider == "groq":
        return LLMProviderInfo(
            base_url=os.getenv("GROQ_API_URL"),
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL_NAME")
        )
    elif provider == "fireworks":
        return LLMProviderInfo(
            base_url=os.getenv("FIREWORKS_API_URL"),
            api_key=os.getenv("FIREWORKS_API_KEY"),
            model_name=os.getenv("FIREWORKS_MODEL_NAME")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Example usage and testing
if __name__ == "__main__":
    
    
    # Initialize the LLM (example with OpenAI)
    # In practice, load these from environment variables
    llm_provider_info   = get_llm_provider("fireworks")
    llm = RagLLM.create(
        base_url=llm_provider_info.base_url,
        api_key=llm_provider_info.api_key,
        model_name=llm_provider_info.model_name
    )
    
    # Print model information
    llm.print_model_info()
    
    # Example 1: Simple generation
    print("Example 1: Simple Generation")
    print("-" * 50)
    response = llm.generate("How to evaluate RAG applications?")
    print(response)
    
    # Example 2: RAG with context
    print("\nExample 2: RAG with Context")
    print("-" * 50)
    context_docs = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Supervised learning requires labeled training data."
    ]
    response = llm.generate_with_context(
         query="What is machine learning? and how does it related to deep learning?",
         context_documents=context_docs
    )
    print(response)
    
    # Example 3: Conversation with history
    print("\nExample 3: Conversation History")
    print("-" * 50)
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "What are its main features?"}
    ]
    # response = llm.generate_with_history(conversation)
    # print(response)
