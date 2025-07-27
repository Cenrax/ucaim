"""Database storage implementation for CAIM framework."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .memory_store import MemoryStore
from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from ..core.config import CAIMConfig
from ..core.exceptions import StorageException


logger = logging.getLogger(__name__)

# Database models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class MemoryRecord(Base):
        __tablename__ = 'memories'
        
        id = Column(String, primary_key=True)
        content = Column(Text, nullable=False)
        memory_type = Column(String, nullable=False)
        importance = Column(Float, nullable=False)
        session_id = Column(String, nullable=False, index=True)
        timestamp = Column(DateTime, nullable=False, index=True)
        last_accessed = Column(DateTime)
        access_count = Column(Integer, default=0)
        decay_factor = Column(Float, default=1.0)
        consolidation_level = Column(Integer, default=0)
        tags = Column(JSON)
        memory_metadata = Column(JSON)
        
        def to_memory(self) -> Memory:
            """Convert database record to Memory object."""
            return Memory(
                id=self.id,
                content=self.content,
                memory_type=MemoryType(self.memory_type),
                importance=MemoryImportance(self.importance),
                session_id=self.session_id,
                timestamp=self.timestamp,
                last_accessed=self.last_accessed,
                access_count=self.access_count,
                decay_factor=self.decay_factor,
                consolidation_level=self.consolidation_level,
                tags=self.tags or [],
                metadata=self.memory_metadata or {}
            )


class DatabaseStore(MemoryStore):
    """PostgreSQL/SQLite database implementation of MemoryStore."""
    
    def __init__(self, config: CAIMConfig, database_url: Optional[str] = None):
        super().__init__(config)
        
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for DatabaseStore")
        
        self.database_url = database_url or config.database_url
        if not self.database_url:
            # Default to SQLite for development
            self.database_url = "sqlite:///caim_memories.db"
        
        self.engine = None
        self.SessionLocal = None
        
        logger.info(f"Initialized DatabaseStore with URL: {self.database_url}")
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise StorageException(f"Database initialization failed: {e}")
    
    def _get_session(self) -> Session:
        """Get database session."""
        if not self.SessionLocal:
            raise StorageException("Database not initialized")
        return self.SessionLocal()
    
    async def store(self, memory: Memory) -> Memory:
        """Store a memory in the database."""
        try:
            session = self._get_session()
            
            try:
                # Convert Memory to database record
                record = MemoryRecord(
                    id=memory.id,
                    content=memory.content,
                    memory_type=memory.memory_type.value,
                    importance=memory.importance.value,
                    session_id=memory.session_id,
                    timestamp=memory.timestamp,
                    last_accessed=memory.last_accessed,
                    access_count=memory.access_count,
                    decay_factor=memory.decay_factor,
                    consolidation_level=memory.consolidation_level,
                    tags=memory.tags,
                    memory_metadata=memory.metadata
                )
                
                # Check if memory already exists
                existing = session.query(MemoryRecord).filter(MemoryRecord.id == memory.id).first()
                
                if existing:
                    # Update existing record
                    for attr in ['content', 'memory_type', 'importance', 'session_id', 
                                'timestamp', 'last_accessed', 'access_count', 'decay_factor',
                                'consolidation_level', 'tags', 'memory_metadata']:
                        setattr(existing, attr, getattr(record, attr))
                else:
                    # Add new record
                    session.add(record)
                
                session.commit()
                
                logger.debug(f"Stored memory {memory.id} in database")
                return memory
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error storing memory in database: {e}")
            raise StorageException(f"Failed to store memory: {e}")
    
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        try:
            session = self._get_session()
            
            try:
                record = session.query(MemoryRecord).filter(MemoryRecord.id == memory_id).first()
                
                if record:
                    # Update access information
                    record.last_accessed = datetime.utcnow()
                    record.access_count += 1
                    session.commit()
                    
                    return record.to_memory()
                
                return None
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def update(self, memory: Memory) -> Memory:
        """Update an existing memory."""
        try:
            session = self._get_session()
            
            try:
                record = session.query(MemoryRecord).filter(MemoryRecord.id == memory.id).first()
                
                if not record:
                    raise StorageException(f"Memory {memory.id} not found")
                
                # Update record
                record.content = memory.content
                record.memory_type = memory.memory_type.value
                record.importance = memory.importance.value
                record.session_id = memory.session_id
                record.timestamp = memory.timestamp
                record.last_accessed = memory.last_accessed
                record.access_count = memory.access_count
                record.decay_factor = memory.decay_factor
                record.consolidation_level = memory.consolidation_level
                record.tags = memory.tags
                record.memory_metadata = memory.metadata
                
                session.commit()
                
                logger.debug(f"Updated memory {memory.id}")
                return memory
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise StorageException(f"Failed to update memory: {e}")
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            session = self._get_session()
            
            try:
                record = session.query(MemoryRecord).filter(MemoryRecord.id == memory_id).first()
                
                if record:
                    session.delete(record)
                    session.commit()
                    logger.debug(f"Deleted memory {memory_id}")
                    return True
                
                return False
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def get_memories_by_session(self, session_id: str) -> List[Memory]:
        """Get all memories for a specific session."""
        try:
            session = self._get_session()
            
            try:
                records = session.query(MemoryRecord)\
                    .filter(MemoryRecord.session_id == session_id)\
                    .order_by(MemoryRecord.timestamp)\
                    .all()
                
                memories = [record.to_memory() for record in records]
                return memories
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting memories for session {session_id}: {e}")
            return []
    
    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        """Get all memories with optional limit."""
        try:
            session = self._get_session()
            
            try:
                query = session.query(MemoryRecord).order_by(MemoryRecord.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                
                records = query.all()
                memories = [record.to_memory() for record in records]
                return memories
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            return []
    
    async def search_memories(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search memories by content and filters."""
        try:
            session = self._get_session()
            
            try:
                db_query = session.query(MemoryRecord)\
                    .filter(MemoryRecord.content.contains(query))
                
                # Apply filters
                if filters:
                    if "session_id" in filters:
                        db_query = db_query.filter(MemoryRecord.session_id == filters["session_id"])
                    
                    if "memory_type" in filters:
                        memory_type = filters["memory_type"]
                        if isinstance(memory_type, MemoryType):
                            memory_type = memory_type.value
                        db_query = db_query.filter(MemoryRecord.memory_type == memory_type)
                    
                    if "importance" in filters:
                        importance = filters["importance"]
                        if isinstance(importance, MemoryImportance):
                            importance = importance.value
                        db_query = db_query.filter(MemoryRecord.importance >= importance)
                    
                    if "start_date" in filters:
                        db_query = db_query.filter(MemoryRecord.timestamp >= filters["start_date"])
                    
                    if "end_date" in filters:
                        db_query = db_query.filter(MemoryRecord.timestamp <= filters["end_date"])
                
                # Order by importance and recency
                records = db_query\
                    .order_by(MemoryRecord.importance.desc(), MemoryRecord.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                memories = [record.to_memory() for record in records]
                return memories
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def update_importance(
        self,
        memory_id: str,
        importance: MemoryImportance
    ) -> bool:
        """Update memory importance level."""
        try:
            session = self._get_session()
            
            try:
                record = session.query(MemoryRecord).filter(MemoryRecord.id == memory_id).first()
                
                if record:
                    record.importance = importance.value
                    session.commit()
                    logger.debug(f"Updated importance for memory {memory_id} to {importance}")
                    return True
                
                return False
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            return False
    
    async def cleanup_expired_memories(self, max_age_days: int = 365) -> int:
        """Remove expired memories and return count removed."""
        try:
            session = self._get_session()
            
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
                
                # Don't remove critical memories or inductive thoughts
                records_to_delete = session.query(MemoryRecord)\
                    .filter(MemoryRecord.timestamp < cutoff_date)\
                    .filter(MemoryRecord.importance < MemoryImportance.CRITICAL.value)\
                    .filter(MemoryRecord.memory_type != MemoryType.INDUCTIVE_THOUGHT.value)\
                    .all()
                
                deleted_count = 0
                for record in records_to_delete:
                    session.delete(record)
                    deleted_count += 1
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_count} expired memories from database")
                return deleted_count
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error cleaning up expired memories: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database storage statistics."""
        try:
            session = self._get_session()
            
            try:
                # Total count
                total_count = session.query(MemoryRecord).count()
                
                # Memory type counts
                type_counts = {}
                for memory_type in MemoryType:
                    count = session.query(MemoryRecord)\
                        .filter(MemoryRecord.memory_type == memory_type.value)\
                        .count()
                    type_counts[memory_type.value] = count
                
                # Session counts
                session_count = session.query(MemoryRecord.session_id)\
                    .distinct()\
                    .count()
                
                # Recent activity (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(days=1)
                recent_count = session.query(MemoryRecord)\
                    .filter(MemoryRecord.timestamp >= recent_cutoff)\
                    .count()
                
                return {
                    "total_memories": total_count,
                    "memory_types": type_counts,
                    "unique_sessions": session_count,
                    "recent_memories_24h": recent_count,
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def close(self) -> None:
        """Close database connections."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")