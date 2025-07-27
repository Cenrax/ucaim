"""Vector storage for memory embeddings."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..memory.memory_types import Memory
from ..core.config import CAIMConfig
from ..core.exceptions import StorageException


logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    
    def __init__(self, config: CAIMConfig):
        self.config = config
        self.collection_name = "caim_memories"
    
    @abstractmethod
    async def store_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store an embedding with metadata."""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings and return (memory_id, similarity_score) pairs."""
        pass
    
    @abstractmethod
    async def delete_embedding(self, memory_id: str) -> bool:
        """Delete an embedding by memory ID."""
        pass
    
    @abstractmethod
    async def update_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing embedding."""
        pass
    
    @abstractmethod
    async def get_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Get embedding by memory ID."""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of VectorStore."""
    
    def __init__(self, config: CAIMConfig, persist_directory: Optional[str] = None):
        super().__init__(config)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required but not installed")
        
        self.persist_directory = persist_directory or "./chroma_db"
        
        # Initialize ChromaDB client
        settings = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CAIM memory embeddings"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    async def store_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store an embedding with metadata."""
        try:
            # ChromaDB requires string values in metadata
            processed_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        processed_metadata[key] = str(value)
                    elif value is not None:
                        processed_metadata[key] = str(value)
            
            self.collection.add(
                embeddings=[embedding],
                metadatas=[processed_metadata],
                ids=[memory_id]
            )
            
            logger.debug(f"Stored embedding for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for {memory_id}: {e}")
            return False
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        try:
            # Convert filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if value is not None:
                        where_clause[key] = str(value)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            # Convert results to (memory_id, similarity_score) pairs
            similar_memories = []
            ids = results['ids'][0]
            distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
            
            for memory_id, distance in zip(ids, distances):
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1.0 - distance if distance <= 1.0 else 0.0
                similar_memories.append((memory_id, similarity))
            
            return similar_memories
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    async def delete_embedding(self, memory_id: str) -> bool:
        """Delete an embedding by memory ID."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Deleted embedding for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding for {memory_id}: {e}")
            return False
    
    async def update_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing embedding."""
        try:
            # ChromaDB doesn't have direct update, so we delete and re-add
            await self.delete_embedding(memory_id)
            return await self.store_embedding(memory_id, embedding, metadata)
            
        except Exception as e:
            logger.error(f"Error updating embedding for {memory_id}: {e}")
            return False
    
    async def get_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Get embedding by memory ID."""
        try:
            results = self.collection.get(
                ids=[memory_id],
                include=["embeddings"]
            )
            
            if results['embeddings'] and len(results['embeddings']) > 0:
                return results['embeddings'][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting embedding for {memory_id}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_embeddings": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore for testing."""
    
    def __init__(self, config: CAIMConfig):
        super().__init__(config)
        self.embeddings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized InMemoryVectorStore")
    
    async def store_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store an embedding with metadata."""
        try:
            self.embeddings[memory_id] = embedding
            self.metadata[memory_id] = metadata or {}
            logger.debug(f"Stored embedding for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for {memory_id}: {e}")
            return False
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings using cosine similarity."""
        try:
            if not self.embeddings:
                return []
            
            query_vector = np.array(query_embedding)
            similarities = []
            
            for memory_id, embedding in self.embeddings.items():
                # Apply filters
                if filters:
                    memory_metadata = self.metadata.get(memory_id, {})
                    skip = False
                    for key, value in filters.items():
                        if key in memory_metadata and str(memory_metadata[key]) != str(value):
                            skip = True
                            break
                    if skip:
                        continue
                
                # Calculate cosine similarity
                embedding_vector = np.array(embedding)
                
                # Normalize vectors
                query_norm = np.linalg.norm(query_vector)
                embedding_norm = np.linalg.norm(embedding_vector)
                
                if query_norm == 0 or embedding_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(query_vector, embedding_vector) / (query_norm * embedding_norm)
                
                similarities.append((memory_id, float(similarity)))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    async def delete_embedding(self, memory_id: str) -> bool:
        """Delete an embedding by memory ID."""
        try:
            if memory_id in self.embeddings:
                del self.embeddings[memory_id]
            if memory_id in self.metadata:
                del self.metadata[memory_id]
            
            logger.debug(f"Deleted embedding for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding for {memory_id}: {e}")
            return False
    
    async def update_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing embedding."""
        return await self.store_embedding(memory_id, embedding, metadata)
    
    async def get_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Get embedding by memory ID."""
        return self.embeddings.get(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_embeddings": len(self.embeddings),
            "total_metadata": len(self.metadata)
        }