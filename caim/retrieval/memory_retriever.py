"""Memory retrieval with filtering and ranking."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from ..storage.memory_store import MemoryStore
from ..storage.vector_store import VectorStore
from .embedding_generator import EmbeddingGenerator
from .similarity_calculator import SimilarityCalculator
from ..core.config import CAIMConfig
from ..core.exceptions import RetrievalException


logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Memory Retrieval module responsible for fetching correct information from LTM.
    Uses filtering mechanisms to prioritize data based on contextual and temporal relevance.
    """
    
    def __init__(
        self,
        config: CAIMConfig,
        memory_store: MemoryStore,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        self.config = config
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.similarity_calculator = SimilarityCalculator()
        
        self.retrieval_cache: Dict[str, Tuple[List[Memory], datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Initialized MemoryRetriever")
    
    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        importance_threshold: Optional[MemoryImportance] = None,
        limit: int = 10,
        use_cache: bool = True
    ) -> List[Memory]:
        """
        Retrieve relevant memories based on query and filters.
        
        Args:
            query: Search query
            session_id: Optional session filter
            context: Additional context for retrieval
            memory_types: Filter by memory types
            importance_threshold: Minimum importance level
            limit: Maximum number of memories to return
            use_cache: Whether to use retrieval cache
            
        Returns:
            List of relevant memories ranked by relevance
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(
                query, session_id, context, memory_types, importance_threshold, limit
            )
            
            # Check cache
            if use_cache and cache_key in self.retrieval_cache:
                cached_memories, cache_time = self.retrieval_cache[cache_key]
                if (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl:
                    logger.debug(f"Retrieved {len(cached_memories)} memories from cache")
                    return cached_memories
            
            # Combine query and context
            full_query = query
            if context:
                full_query = f"{query} {context}"
            
            # Get semantic matches using vector search
            semantic_memories = await self._semantic_retrieval(
                full_query, session_id, memory_types, limit * 2
            )
            
            # Get keyword matches using text search
            keyword_memories = await self._keyword_retrieval(
                query, session_id, memory_types, limit
            )
            
            # Combine and deduplicate results
            all_memories = self._merge_results(semantic_memories, keyword_memories)
            
            # Apply importance filter
            if importance_threshold:
                all_memories = [
                    m for m in all_memories
                    if m.importance.value >= importance_threshold.value
                ]
            
            # Rank memories by relevance
            ranked_memories = await self._rank_memories(
                all_memories, query, context, session_id
            )
            
            # Apply final limit
            final_memories = ranked_memories[:limit]
            
            # Cache results
            if use_cache:
                self.retrieval_cache[cache_key] = (final_memories, datetime.utcnow())
            
            logger.info(f"Retrieved {len(final_memories)} relevant memories")
            return final_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            raise RetrievalException(f"Failed to retrieve memories: {e}")
    
    async def _semantic_retrieval(
        self,
        query: str,
        session_id: Optional[str],
        memory_types: Optional[List[MemoryType]],
        limit: int
    ) -> List[Memory]:
        """Retrieve memories using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Prepare filters for vector search
            filters = {}
            if session_id:
                filters["session_id"] = session_id
            if memory_types:
                # For now, we'll filter after retrieval since vector stores may not support complex filters
                pass
            
            # Search for similar embeddings
            similar_ids = await self.vector_store.search_similar(
                query_embedding, limit=limit, filters=filters
            )
            
            # Get memory objects
            memories = []
            for memory_id, similarity_score in similar_ids:
                memory = await self.memory_store.get(memory_id)
                if memory:
                    # Add similarity score to metadata
                    memory.metadata["semantic_similarity"] = similarity_score
                    
                    # Apply memory type filter
                    if memory_types and memory.memory_type not in memory_types:
                        continue
                    
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    async def _keyword_retrieval(
        self,
        query: str,
        session_id: Optional[str],
        memory_types: Optional[List[MemoryType]],
        limit: int
    ) -> List[Memory]:
        """Retrieve memories using keyword search."""
        try:
            # Prepare filters
            filters = {}
            if session_id:
                filters["session_id"] = session_id
            
            # Search memories by content
            memories = await self.memory_store.search_memories(
                query=query,
                filters=filters,
                limit=limit
            )
            
            # Apply memory type filter
            if memory_types:
                memories = [m for m in memories if m.memory_type in memory_types]
            
            # Add keyword match score
            for memory in memories:
                score = self._calculate_keyword_score(query, memory.content)
                memory.metadata["keyword_score"] = score
            
            return memories
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {e}")
            return []
    
    def _merge_results(
        self,
        semantic_memories: List[Memory],
        keyword_memories: List[Memory]
    ) -> List[Memory]:
        """Merge and deduplicate semantic and keyword results."""
        try:
            seen_ids = set()
            merged_memories = []
            
            # Add semantic memories first (usually higher quality)
            for memory in semantic_memories:
                if memory.id not in seen_ids:
                    merged_memories.append(memory)
                    seen_ids.add(memory.id)
            
            # Add keyword memories that weren't already included
            for memory in keyword_memories:
                if memory.id not in seen_ids:
                    merged_memories.append(memory)
                    seen_ids.add(memory.id)
            
            return merged_memories
            
        except Exception as e:
            logger.error(f"Error merging results: {e}")
            return semantic_memories + keyword_memories
    
    async def _rank_memories(
        self,
        memories: List[Memory],
        query: str,
        context: Optional[str],
        session_id: Optional[str]
    ) -> List[Memory]:
        """Rank memories by relevance using multiple factors."""
        try:
            current_time = datetime.utcnow()
            
            for memory in memories:
                # Calculate composite relevance score
                relevance_score = await self._calculate_relevance_score(
                    memory, query, context, session_id, current_time
                )
                memory.metadata["relevance_score"] = relevance_score
            
            # Sort by relevance score (highest first)
            ranked_memories = sorted(
                memories,
                key=lambda m: m.metadata.get("relevance_score", 0),
                reverse=True
            )
            
            return ranked_memories
            
        except Exception as e:
            logger.error(f"Error ranking memories: {e}")
            return memories
    
    async def _calculate_relevance_score(
        self,
        memory: Memory,
        query: str,
        context: Optional[str],
        session_id: Optional[str],
        current_time: datetime
    ) -> float:
        """Calculate comprehensive relevance score for a memory."""
        try:
            # Base memory relevance (importance, recency, access frequency)
            base_score = memory.calculate_relevance_score(current_time)
            
            # Semantic similarity score
            semantic_score = memory.metadata.get("semantic_similarity", 0.0)
            
            # Keyword match score
            keyword_score = memory.metadata.get("keyword_score", 0.0)
            
            # Session relevance (higher score for same session)
            session_score = 1.0 if memory.session_id == session_id else 0.5
            
            # Context relevance
            context_score = 0.0
            if context:
                context_score = await self.calculate_similarity(memory.content, context)
            
            # Memory type bonus
            type_bonus = self._get_memory_type_bonus(memory.memory_type)
            
            # Weighted combination
            relevance_score = (
                base_score * 0.3 +
                semantic_score * 0.3 +
                keyword_score * 0.2 +
                session_score * 0.1 +
                context_score * 0.05 +
                type_bonus * 0.05
            )
            
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword match score."""
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Error calculating keyword score: {e}")
            return 0.0
    
    def _get_memory_type_bonus(self, memory_type: MemoryType) -> float:
        """Get bonus score based on memory type."""
        type_bonuses = {
            MemoryType.INDUCTIVE_THOUGHT: 0.3,
            MemoryType.FACTUAL: 0.2,
            MemoryType.EMOTIONAL: 0.15,
            MemoryType.EPISODIC: 0.1,
            MemoryType.SEMANTIC: 0.1,
            MemoryType.PROCEDURAL: 0.05,
            MemoryType.CONVERSATIONAL: 0.0
        }
        return type_bonuses.get(memory_type, 0.0)
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            return await self.similarity_calculator.calculate_text_similarity(text1, text2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def calculate_similarity_vectors(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """Calculate similarity between two vectors."""
        try:
            return await self.similarity_calculator.calculate_vector_similarity(vector1, vector2)
        except Exception as e:
            logger.error(f"Error calculating vector similarity: {e}")
            return 0.0
    
    def _create_cache_key(
        self,
        query: str,
        session_id: Optional[str],
        context: Optional[str],
        memory_types: Optional[List[MemoryType]],
        importance_threshold: Optional[MemoryImportance],
        limit: int
    ) -> str:
        """Create cache key for retrieval parameters."""
        key_parts = [
            f"query:{query}",
            f"session:{session_id or 'none'}",
            f"context:{context or 'none'}",
            f"types:{[t.value for t in memory_types] if memory_types else 'none'}",
            f"importance:{importance_threshold.value if importance_threshold else 'none'}",
            f"limit:{limit}"
        ]
        return "|".join(key_parts)
    
    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.retrieval_cache.clear()
        logger.info("Cleared retrieval cache")
    
    async def get_related_memories(
        self,
        memory: Memory,
        limit: int = 5
    ) -> List[Memory]:
        """Get memories related to a given memory."""
        try:
            if not memory.embedding:
                # Generate embedding if not present
                memory.embedding = await self.embedding_generator.generate_embedding(memory.content)
            
            # Find similar memories
            similar_ids = await self.vector_store.search_similar(
                memory.embedding,
                limit=limit + 1  # +1 to exclude the original memory
            )
            
            related_memories = []
            for memory_id, similarity_score in similar_ids:
                if memory_id != memory.id:  # Exclude the original memory
                    related_memory = await self.memory_store.get(memory_id)
                    if related_memory:
                        related_memory.metadata["similarity_to_source"] = similarity_score
                        related_memories.append(related_memory)
            
            return related_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting related memories: {e}")
            return []
    
    async def get_memory_timeline_context(
        self,
        memory: Memory,
        context_window: int = 5
    ) -> List[Memory]:
        """Get memories from the same session around the same time."""
        try:
            session_memories = await self.memory_store.get_memories_by_session(memory.session_id)
            
            # Sort by timestamp
            session_memories.sort(key=lambda m: m.timestamp)
            
            # Find the position of the target memory
            target_index = -1
            for i, mem in enumerate(session_memories):
                if mem.id == memory.id:
                    target_index = i
                    break
            
            if target_index == -1:
                return []
            
            # Get context window around the target memory
            start_idx = max(0, target_index - context_window // 2)
            end_idx = min(len(session_memories), target_index + context_window // 2 + 1)
            
            context_memories = session_memories[start_idx:end_idx]
            
            # Remove the target memory itself
            context_memories = [m for m in context_memories if m.id != memory.id]
            
            return context_memories
            
        except Exception as e:
            logger.error(f"Error getting timeline context: {e}")
            return []