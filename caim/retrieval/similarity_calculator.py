"""Similarity calculation utilities for memory retrieval."""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..core.exceptions import ModelException


logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculator for various similarity metrics."""
    
    def __init__(self):
        logger.info("Initialized SimilarityCalculator")
    
    async def calculate_vector_similarity(
        self,
        vector1: List[float],
        vector2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metric: Similarity metric ('cosine', 'euclidean', 'dot_product')
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if len(vector1) != len(vector2):
                logger.warning(f"Vector dimension mismatch: {len(vector1)} vs {len(vector2)}")
                return 0.0
            
            if metric == "cosine":
                return await self._cosine_similarity(vector1, vector2)
            elif metric == "euclidean":
                return await self._euclidean_similarity(vector1, vector2)
            elif metric == "dot_product":
                return await self._dot_product_similarity(vector1, vector2)
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Error calculating vector similarity: {e}")
            return 0.0
    
    async def calculate_text_similarity(
        self,
        text1: str,
        text2: str,
        method: str = "jaccard"
    ) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('jaccard', 'overlap', 'levenshtein')
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if method == "jaccard":
                return await self._jaccard_similarity(text1, text2)
            elif method == "overlap":
                return await self._overlap_similarity(text1, text2)
            elif method == "levenshtein":
                return await self._levenshtein_similarity(text1, text2)
            else:
                raise ValueError(f"Unknown text similarity method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    async def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if NUMPY_AVAILABLE:
                v1 = np.array(vector1)
                v2 = np.array(vector2)
                
                # Calculate norms
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                # Calculate cosine similarity
                similarity = np.dot(v1, v2) / (norm1 * norm2)
                return float(max(0.0, min(1.0, similarity)))
            else:
                # Pure Python implementation
                dot_product = sum(a * b for a, b in zip(vector1, vector2))
                
                norm1 = math.sqrt(sum(a * a for a in vector1))
                norm2 = math.sqrt(sum(a * a for a in vector2))
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return max(0.0, min(1.0, similarity))
                
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _euclidean_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate euclidean similarity (inverse of distance) between two vectors."""
        try:
            if NUMPY_AVAILABLE:
                v1 = np.array(vector1)
                v2 = np.array(vector2)
                distance = np.linalg.norm(v1 - v2)
            else:
                # Pure Python implementation
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
            
            # Convert distance to similarity (0 distance = 1 similarity)
            # Use exponential decay to map distance to [0, 1]
            similarity = math.exp(-distance / len(vector1))
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.error(f"Error calculating euclidean similarity: {e}")
            return 0.0
    
    async def _dot_product_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate normalized dot product similarity between two vectors."""
        try:
            if NUMPY_AVAILABLE:
                v1 = np.array(vector1)
                v2 = np.array(vector2)
                dot_product = np.dot(v1, v2)
            else:
                dot_product = sum(a * b for a, b in zip(vector1, vector2))
            
            # Normalize by vector length for fair comparison
            max_possible = len(vector1)
            normalized_similarity = dot_product / max_possible
            
            return float(max(0.0, min(1.0, normalized_similarity)))
            
        except Exception as e:
            logger.error(f"Error calculating dot product similarity: {e}")
            return 0.0
    
    async def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        try:
            # Tokenize texts
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if not tokens1 and not tokens2:
                return 1.0
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"Error calculating Jaccard similarity: {e}")
            return 0.0
    
    async def _overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate overlap coefficient between two texts."""
        try:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if not tokens1 and not tokens2:
                return 1.0
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Calculate overlap coefficient
            intersection = len(tokens1.intersection(tokens2))
            min_size = min(len(tokens1), len(tokens2))
            
            if min_size == 0:
                return 0.0
            
            return intersection / min_size
            
        except Exception as e:
            logger.error(f"Error calculating overlap similarity: {e}")
            return 0.0
    
    async def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein similarity between two texts."""
        try:
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(text1, text2)
            
            # Normalize by maximum possible distance
            max_length = max(len(text1), len(text2))
            
            if max_length == 0:
                return 1.0
            
            # Convert distance to similarity
            similarity = 1.0 - (distance / max_length)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating Levenshtein similarity: {e}")
            return 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    async def calculate_semantic_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        text1: str,
        text2: str,
        embedding_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> float:
        """
        Calculate combined semantic similarity using both embeddings and text.
        
        Args:
            embedding1: First text embedding
            embedding2: Second text embedding
            text1: First text
            text2: Second text
            embedding_weight: Weight for embedding similarity
            text_weight: Weight for text similarity
            
        Returns:
            Combined similarity score
        """
        try:
            # Calculate embedding similarity
            embedding_sim = await self.calculate_vector_similarity(
                embedding1, embedding2, metric="cosine"
            )
            
            # Calculate text similarity
            text_sim = await self.calculate_text_similarity(
                text1, text2, method="jaccard"
            )
            
            # Weighted combination
            combined_similarity = (
                embedding_sim * embedding_weight +
                text_sim * text_weight
            )
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def calculate_contextual_similarity(
        self,
        memory1_content: str,
        memory2_content: str,
        memory1_context: Dict[str, Any],
        memory2_context: Dict[str, Any],
        content_weight: float = 0.6,
        context_weight: float = 0.4
    ) -> float:
        """
        Calculate similarity considering both content and context.
        
        Args:
            memory1_content: Content of first memory
            memory2_content: Content of second memory
            memory1_context: Context metadata of first memory
            memory2_context: Context metadata of second memory
            content_weight: Weight for content similarity
            context_weight: Weight for context similarity
            
        Returns:
            Combined similarity score
        """
        try:
            # Calculate content similarity
            content_sim = await self.calculate_text_similarity(
                memory1_content, memory2_content
            )
            
            # Calculate context similarity
            context_sim = self._calculate_context_similarity(
                memory1_context, memory2_context
            )
            
            # Weighted combination
            combined_similarity = (
                content_sim * content_weight +
                context_sim * context_weight
            )
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating contextual similarity: {e}")
            return 0.0
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between context dictionaries."""
        try:
            if not context1 and not context2:
                return 1.0
            
            if not context1 or not context2:
                return 0.0
            
            # Get common keys
            keys1 = set(context1.keys())
            keys2 = set(context2.keys())
            common_keys = keys1.intersection(keys2)
            
            if not common_keys:
                return 0.0
            
            # Calculate similarity for common keys
            similarities = []
            for key in common_keys:
                val1 = str(context1[key])
                val2 = str(context2[key])
                
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    # Use text similarity for different values
                    sim = len(set(val1.lower().split()).intersection(set(val2.lower().split())))
                    sim = sim / max(len(val1.split()), len(val2.split())) if val1 or val2 else 0.0
                    similarities.append(sim)
            
            # Average similarity across common keys
            avg_similarity = sum(similarities) / len(similarities)
            
            # Penalize for missing keys
            key_coverage = len(common_keys) / max(len(keys1), len(keys2))
            
            return avg_similarity * key_coverage
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    async def batch_similarity_calculation(
        self,
        query_vector: List[float],
        target_vectors: List[List[float]],
        metric: str = "cosine"
    ) -> List[float]:
        """
        Calculate similarity between a query vector and multiple target vectors.
        
        Args:
            query_vector: Query vector
            target_vectors: List of target vectors
            metric: Similarity metric
            
        Returns:
            List of similarity scores
        """
        try:
            similarities = []
            
            for target_vector in target_vectors:
                similarity = await self.calculate_vector_similarity(
                    query_vector, target_vector, metric
                )
                similarities.append(similarity)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            return [0.0] * len(target_vectors)