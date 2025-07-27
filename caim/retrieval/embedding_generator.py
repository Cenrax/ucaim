"""Embedding generation for memory content."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import asyncio

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.config import CAIMConfig
from ..core.exceptions import ModelException


logger = logging.getLogger(__name__)


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generation."""
    
    def __init__(self, config: CAIMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass


class SentenceTransformerEmbedding(EmbeddingGenerator):
    """Sentence Transformers implementation for embeddings."""
    
    def __init__(self, config: CAIMConfig, model_name: Optional[str] = None):
        super().__init__(config)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required but not installed")
        
        self.model_name = model_name or config.embedding_model
        self.model = None
        self._embedding_dim = None
        
        logger.info(f"Initializing SentenceTransformer with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                # Get embedding dimension
                sample_embedding = self.model.encode(["test"])
                self._embedding_dim = len(sample_embedding[0])
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                raise ModelException(f"Failed to load model {self.model_name}: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            self._load_model()
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode([text])[0]
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise ModelException(f"Failed to generate embedding: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            self._load_model()
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts)
            )
            
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ModelException(f"Failed to generate embeddings: {e}")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim


class OpenAIEmbedding(EmbeddingGenerator):
    """OpenAI implementation for embeddings."""
    
    def __init__(self, config: CAIMConfig, model_name: str = "text-embedding-ada-002"):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required but not installed")
        
        self.model_name = model_name
        self.client = None
        self._embedding_dim = 1536  # Default for ada-002
        
        if not config.openai_api_key:
            raise ModelException("OpenAI API key is required")
        
        logger.info(f"Initializing OpenAI embeddings with model: {self.model_name}")
    
    def _get_client(self) -> openai.OpenAI:
        """Get OpenAI client."""
        if self.client is None:
            self.client = openai.OpenAI(api_key=self.config.openai_api_key)
        return self.client
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            client = self._get_client()
            
            # Make API call
            response = client.embeddings.create(
                input=text,
                model=self.model_name
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise ModelException(f"Failed to generate OpenAI embedding: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            client = self._get_client()
            
            # OpenAI supports batch requests
            response = client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise ModelException(f"Failed to generate OpenAI embeddings: {e}")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._embedding_dim


class HuggingFaceEmbedding(EmbeddingGenerator):
    """HuggingFace Transformers implementation for embeddings."""
    
    def __init__(self, config: CAIMConfig, model_name: Optional[str] = None):
        super().__init__(config)
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModel = AutoModel
            self.torch = torch
        except ImportError:
            raise ImportError("transformers and torch are required but not installed")
        
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        self._embedding_dim = None
        
        logger.info(f"Initializing HuggingFace embeddings with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self.model is None:
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
                self.model = self.AutoModel.from_pretrained(self.model_name)
                
                # Get embedding dimension
                with self.torch.no_grad():
                    inputs = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                    outputs = self.model(**inputs)
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
                    self._embedding_dim = pooled_output.shape[1]
                
                logger.info(f"Loaded HuggingFace model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading HuggingFace model: {e}")
                raise ModelException(f"Failed to load model {self.model_name}: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            self._load_model()
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._encode_text,
                text
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating HuggingFace embedding: {e}")
            raise ModelException(f"Failed to generate embedding: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            self._load_model()
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._encode_texts,
                texts
            )
            
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise ModelException(f"Failed to generate embeddings: {e}")
    
    def _encode_text(self, text: str):
        """Encode single text to embedding."""
        with self.torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embedding
    
    def _encode_texts(self, texts: List[str]):
        """Encode multiple texts to embeddings."""
        with self.torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim


def create_embedding_generator(config: CAIMConfig, provider: str = "sentence_transformers") -> EmbeddingGenerator:
    """Factory function to create embedding generators."""
    
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedding(config)
    elif provider == "openai":
        return OpenAIEmbedding(config)
    elif provider == "huggingface":
        return HuggingFaceEmbedding(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Simple embedding for testing without dependencies
class SimpleEmbedding(EmbeddingGenerator):
    """Simple TF-IDF based embedding for testing."""
    
    def __init__(self, config: CAIMConfig):
        super().__init__(config)
        self.vocabulary = {}
        self.vocab_size = 1000
        self._embedding_dim = 300
        
        logger.info("Initialized SimpleEmbedding (TF-IDF based)")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for text."""
        try:
            words = text.lower().split()
            
            # Simple TF-IDF approximation
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Create embedding vector
            embedding = [0.0] * self._embedding_dim
            
            for i, word in enumerate(word_counts.keys()):
                if i >= self._embedding_dim:
                    break
                # Simple hash-based positioning
                pos = hash(word) % self._embedding_dim
                embedding[pos] = word_counts[word] / len(words)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating simple embedding: {e}")
            return [0.0] * self._embedding_dim
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._embedding_dim