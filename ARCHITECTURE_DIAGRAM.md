# CAIM Framework - Complete Architecture Diagram

## System Overview

```mermaid
graph TB
    %% User Interface Layer
    subgraph "User Interface Layer"
        CLI[CLI Interface<br/>• Interactive Chat<br/>• Benchmarks<br/>• Memory Management]
        Streamlit[Streamlit Demo<br/>• Web Chat<br/>• Memory Insights<br/>• System Status]
        FastAPI[FastAPI Server<br/>• REST Endpoints<br/>• WebSocket Chat<br/>• API Documentation]
        Jupyter[Jupyter Notebooks<br/>• Examples<br/>• Benchmarks<br/>• Tutorials]
    end

    %% Agent Layer
    subgraph "Agent Layer"
        CAIMAgent[CAIM Agent<br/>• Message Processing<br/>• Memory Integration<br/>• Response Generation]
        MultiAgent[Multi-Agent System<br/>• Agent Coordination<br/>• Shared Memory<br/>• Conversation Management]
        AgentFactory[Agent Factory<br/>• Agent Creation<br/>• Configuration<br/>• Lifecycle Management]
    end

    %% Core Framework
    subgraph "Core CAIM Framework"
        MemoryController[Memory Controller<br/>• Access Decisions<br/>• Context Analysis<br/>• Memory Prioritization]
        MemoryRetrieval[Memory Retrieval<br/>• Contextual Filtering<br/>• Temporal Filtering<br/>• Relevance Ranking]
        PostThinking[Post-Thinking<br/>• Memory Consolidation<br/>• Pattern Extraction<br/>• Insight Generation]
        SessionMgmt[Session Management<br/>• Multi-Session Support<br/>• Context Switching<br/>• State Persistence]
    end

    %% Memory System
    subgraph "Memory System"
        STM[Short-Term Memory<br/>• Session Storage<br/>• Recent Context<br/>• Working Memory]
        LTM[Long-Term Memory<br/>• Persistent Storage<br/>• Knowledge Base<br/>• Experience Archive]
        MemoryTypes[Memory Types<br/>• Conversational<br/>• Semantic<br/>• Procedural<br/>• Emotional<br/>• Factual<br/>• Inductive Thoughts]
        MemoryProcessing[Memory Processing<br/>• Consolidation<br/>• Decay/Forgetting<br/>• Importance Scoring<br/>• Access Tracking]
    end

    %% Model Layer
    subgraph "Model Layer"
        HFModels[HuggingFace Models<br/>• Qwen Integration<br/>• Gemma 1B/3B<br/>• Local Inference<br/>• Model Management]
        OpenAI[OpenAI Integration<br/>• GPT Models<br/>• API Management<br/>• Cost Tracking<br/>• Benchmarking]
        ModelAbstraction[Model Abstraction<br/>• Unified Interface<br/>• Response Handling<br/>• Streaming Support<br/>• Error Management]
        Embeddings[Embedding Models<br/>• Sentence Transformers<br/>• Vector Generation<br/>• Similarity Calculation<br/>• Semantic Search]
    end

    %% Storage Layer
    subgraph "Storage Layer"
        InMemory[In-Memory Store<br/>• Fast Access<br/>• Session Cache<br/>• Development Mode]
        VectorStore[Vector Store<br/>• ChromaDB<br/>• Semantic Search<br/>• Similarity Queries<br/>• Embedding Storage]
        DatabaseStore[Database Store<br/>• PostgreSQL<br/>• SQLite Support<br/>• Persistent Storage<br/>• Query Optimization]
        FileSystem[File System<br/>• Configuration Files<br/>• Logs & Analytics<br/>• Export/Import<br/>• Backup Storage]
        CloudStorage[Cloud Storage<br/>• Scalable Storage<br/>• Distributed Access<br/>• Multi-tenant Support]
    end

    %% Support Systems
    subgraph "Support Systems"
        Config[Configuration<br/>• CAIM Config<br/>• Environment Variables<br/>• Model Settings<br/>• System Parameters]
        Logging[Logging & Monitoring<br/>• Structured Logging<br/>• Performance Metrics<br/>• Error Tracking<br/>• System Health]
        Benchmarking[Benchmarking<br/>• Performance Tests<br/>• Memory Efficiency<br/>• Model Comparison<br/>• Quality Metrics]
        Utils[Utilities<br/>• Data Processing<br/>• Helper Functions<br/>• Validation<br/>• Common Tools]
        Exceptions[Exception Handling<br/>• Custom Exceptions<br/>• Error Recovery<br/>• Graceful Degradation<br/>• Fault Tolerance]
    end

    %% Data Flow Connections
    CLI --> CAIMAgent
    Streamlit --> CAIMAgent
    FastAPI --> MultiAgent
    Jupyter --> AgentFactory

    CAIMAgent --> MemoryController
    MultiAgent --> MemoryRetrieval
    AgentFactory --> PostThinking

    MemoryController --> STM
    MemoryController --> LTM
    MemoryRetrieval --> MemoryTypes
    MemoryRetrieval --> MemoryProcessing
    PostThinking --> MemoryProcessing

    STM --> HFModels
    LTM --> OpenAI
    MemoryTypes --> ModelAbstraction
    MemoryProcessing --> Embeddings

    HFModels --> InMemory
    OpenAI --> VectorStore
    ModelAbstraction --> DatabaseStore
    Embeddings --> FileSystem

    InMemory --> Config
    VectorStore --> Logging
    DatabaseStore --> Benchmarking
    FileSystem --> Utils
    CloudStorage --> Exceptions

    %% Styling
    classDef interface fill:#8FBC8F,stroke:#2d5a2d,stroke-width:2px
    classDef agent fill:#4A90E2,stroke:#1a4480,stroke-width:2px
    classDef core fill:#2E86AB,stroke:#1a4d62,stroke-width:2px
    classDef memory fill:#A23B72,stroke:#5a1f3f,stroke-width:2px
    classDef model fill:#F18F01,stroke:#8a5100,stroke-width:2px
    classDef storage fill:#C73E1D,stroke:#6b2110,stroke-width:2px
    classDef support fill:#9B59B6,stroke:#5a3469,stroke-width:2px

    class CLI,Streamlit,FastAPI,Jupyter interface
    class CAIMAgent,MultiAgent,AgentFactory agent
    class MemoryController,MemoryRetrieval,PostThinking,SessionMgmt core
    class STM,LTM,MemoryTypes,MemoryProcessing memory
    class HFModels,OpenAI,ModelAbstraction,Embeddings model
    class InMemory,VectorStore,DatabaseStore,FileSystem,CloudStorage storage
    class Config,Logging,Benchmarking,Utils,Exceptions support
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Interface as CLI/Web Interface
    participant Agent as CAIM Agent
    participant Core as CAIM Framework
    participant Memory as Memory System
    participant Model as AI Models
    participant Storage as Storage Layer

    User->>Interface: Input Message
    Interface->>Agent: Process Message
    Agent->>Core: Forward Input + Context
    
    Core->>Memory: Check Memory Controller
    Memory->>Storage: Retrieve Relevant Memories
    Storage-->>Memory: Return Memories
    Memory-->>Core: Provide Context
    
    Core->>Model: Generate Response with Context
    Model-->>Core: Return AI Response
    
    Core->>Memory: Store New Memory
    Memory->>Storage: Persist Memory
    
    Core->>Memory: Trigger Post-Thinking
    Memory->>Memory: Consolidate & Extract Patterns
    
    Core-->>Agent: Return Enhanced Response
    Agent-->>Interface: Format Response
    Interface-->>User: Display Response + Memory Insights
```

## Memory Consolidation Flow

```mermaid
flowchart TD
    A[New User Input] --> B{Memory Controller Decision}
    B -->|Access STM| C[Retrieve STM Context]
    B -->|Access LTM| D[Retrieve LTM Context]
    B -->|No Memory| E[Process Without Memory]
    
    C --> F[Generate Response with STM]
    D --> G[Generate Response with LTM]
    E --> H[Generate Basic Response]
    
    F --> I[Store in STM]
    G --> I
    H --> I
    
    I --> J{STM Full?}
    J -->|Yes| K[Trigger Consolidation]
    J -->|No| L[Continue Processing]
    
    K --> M[Evaluate Memory Importance]
    M --> N{Important Memory?}
    N -->|Yes| O[Promote to LTM]
    N -->|No| P[Apply Decay/Forget]
    
    O --> Q[Extract Patterns]
    P --> R[Update Memory Weights]
    
    Q --> S[Generate Inductive Thoughts]
    R --> T[Continue Session]
    S --> T
    L --> T
```

## Component Integration

```mermaid
graph LR
    subgraph "Frontend Layer"
        UI[User Interfaces]
    end
    
    subgraph "Application Layer"
        AGENT[CAIM Agents]
    end
    
    subgraph "Business Logic"
        CORE[CAIM Framework Core]
        MEM[Memory System]
    end
    
    subgraph "Service Layer"
        MODELS[AI Models]
        BENCH[Benchmarking]
    end
    
    subgraph "Data Layer"
        STORE[Storage Systems]
        CONFIG[Configuration]
    end
    
    UI --> AGENT
    AGENT --> CORE
    CORE --> MEM
    MEM --> MODELS
    MODELS --> STORE
    STORE --> CONFIG
    
    BENCH --> MODELS
    BENCH --> MEM
    CONFIG --> CORE
```

## Technology Stack

```mermaid
mindmap
  root((CAIM Framework))
    Frontend
      Streamlit
      FastAPI
      CLI
      Jupyter
    Backend
      Python 3.11+
      AsyncIO
      Pydantic
      SQLAlchemy
    AI/ML
      HuggingFace
        Qwen
        Gemma 1B/3B
        Transformers
      OpenAI
        GPT Models
        API Integration
      Embeddings
        SentenceTransformers
        ChromaDB
    Storage
      In-Memory
      PostgreSQL
      SQLite
      Vector Store
      File System
    DevOps
      Docker
      Testing
      Logging
      Monitoring
      Benchmarking
```

## Key Features Overview

- **🧠 Cognitive Memory**: Dual STM/LTM system with intelligent consolidation
- **🤖 Multi-Model Support**: HuggingFace (Qwen, Gemma) + OpenAI integration
- **💾 Flexible Storage**: In-memory, vector, database, and cloud backends
- **🔄 Async Architecture**: Production-ready with proper error handling
- **📊 Real-time Insights**: Memory visualization and performance analytics
- **🚀 Multiple Interfaces**: CLI, Web, API, and Jupyter notebook support
- **⚡ Production Ready**: Comprehensive testing, logging, and monitoring
- **🔧 Extensible**: Modular design for easy customization and scaling