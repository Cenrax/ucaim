# 🧠 CAIM: Cognitive AI Memory Framework

**A production-ready framework for enhanced long-term agent interactions with human-like memory capabilities.**

CAIM (Cognitive AI Memory) implements a comprehensive memory system that enables AI agents to maintain context, learn from interactions, and provide personalized responses across extended conversations.

## ✨ Key Features

- **🧠 Human-like Memory System**: Dual Short-Term Memory (STM) and Long-Term Memory (LTM) architecture
- **🔄 Intelligent Memory Consolidation**: Automatic pattern recognition and memory synthesis
- **🎯 Context-Aware Retrieval**: Semantic and temporal memory filtering for relevant context
- **🤖 Multi-Model Support**: Integration with HuggingFace (Qwen, Gemma) and OpenAI models
- **📊 Comprehensive Benchmarking**: Built-in evaluation metrics and performance analysis
- **🌐 Web Interface**: Streamlit-based demo application for interactive testing
- **⚡ Production Ready**: Async architecture, error handling, and monitoring capabilities

## 🏗️ Architecture

CAIM consists of three core modules inspired by cognitive science:

### 1. Memory Controller
- **Gatekeeper Function**: Determines when memory access is necessary
- **Context Analysis**: Evaluates user input against current conversation state
- **Selective Retrieval**: Prevents information overload through intelligent filtering

### 2. Memory Retrieval
- **Semantic Search**: Vector-based similarity matching for relevant memories
- **Temporal Filtering**: Time-based relevance scoring and decay mechanisms
- **Ontology-Based Tagging**: Structured categorization for efficient retrieval

### 3. Post-Thinking (Memory Consolidation)
- **Pattern Extraction**: Identifies recurring themes and concepts
- **Inductive Thoughts**: Creates high-level summaries from related memories
- **Memory Optimization**: Removes redundant and low-value information

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd caim

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Environment Setup

Create a `.env` file based on `.env.example`:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
# Optional: Add API keys for enhanced functionality
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

### Basic Usage

```python
import asyncio
from caim import CAIMFramework, CAIMAgent, CAIMConfig
from caim.models import create_model

async def basic_example():
    # Initialize framework
    config = CAIMConfig.from_env()
    caim_framework = CAIMFramework(config)
    await caim_framework.initialize()
    
    # Create agent with memory capabilities
    model = create_model("Qwen/Qwen2.5-1.5B-Instruct", config)
    agent = CAIMAgent("MyAgent", config, model, caim_framework)
    await agent.initialize()
    
    # Have a conversation with memory
    session_id = "user_session_001"
    
    response1 = await agent.process_message(
        "Hi, I'm Alice and I love Python programming!", 
        session_id
    )
    print(f"Agent: {response1.content}")
    
    response2 = await agent.process_message(
        "What programming language do I enjoy?", 
        session_id
    )
    print(f"Agent: {response2.content}")
    # Agent remembers Alice likes Python!
    
    # Cleanup
    await agent.shutdown()
    await caim_framework.shutdown()

asyncio.run(basic_example())
```

## 🎛️ Command Line Interface

CAIM includes a comprehensive CLI for easy interaction:

```bash
# Interactive chat mode
python -m caim.cli chat --interactive --model "Qwen/Qwen2.5-1.5B-Instruct"

# Single message
python -m caim.cli chat "Hello, how does CAIM work?"

# Run benchmarks
python -m caim.cli benchmark --full --output results.json

# Memory system inspection
python -m caim.cli memory --session-id my_session

# Get model recommendations
python -m caim.cli recommend --use-case research --max-memory 16

# Launch web demo
python -m caim.cli demo --port 8501

# Initialize configuration
python -m caim.cli init --output ./config/
```

## 🌐 Web Demo

Launch the interactive Streamlit demo:

```bash
# Method 1: Using CLI
python -m caim.cli demo

# Method 2: Direct launch
streamlit run caim/demo/streamlit_app.py
```

The demo provides:
- 💬 **Interactive Chat**: Real-time conversation with memory
- 🧠 **Memory Insights**: Visualization of memory usage and patterns
- 📊 **Benchmarks**: Performance testing and model comparison
- ⚙️ **System Status**: Health monitoring and configuration

## 🤖 Supported Models

### HuggingFace Models
- **Qwen Series**: Qwen2.5-1.5B, Qwen2.5-3B (Recommended)
- **Gemma Series**: Gemma-2B, Gemma-7B
- **Llama Series**: Llama-2-7B, Llama-2-13B
- **Mistral**: Mistral-7B-Instruct

### OpenAI Models (API)
- GPT-3.5-turbo
- GPT-4 series (for benchmarking)

### Model Recommendations

```bash
# Get recommendations based on your requirements
python -m caim.cli recommend --use-case general --max-memory 8 --local-only
```

## 📊 Benchmarking

CAIM includes comprehensive benchmarking capabilities:

### Built-in Benchmarks

1. **Memory Retrieval Accuracy**: Precision, recall, F1-score
2. **Response Coherence**: Context consistency evaluation
3. **Memory Consolidation Quality**: Information preservation metrics
4. **Contextual Awareness**: Cross-conversation context maintenance
5. **System Efficiency**: Response times and resource usage

### Running Benchmarks

```python
from caim.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(config)

# Quick comparison
results = await runner.run_agent_comparison_benchmark(agents)

# Comprehensive evaluation
results = await runner.run_full_benchmark_suite(agents, caim_framework)

# Memory-specific tests
results = await runner.run_memory_benchmark_suite(caim_framework)
```

## 🧪 Examples

Explore the `examples/` directory for detailed usage patterns:

- **`basic_usage.py`**: Fundamental CAIM operations
- **`advanced_usage.py`**: Multi-agent configurations and memory management
- **`benchmarking_example.py`**: Performance evaluation workflows
- **`model_comparison.py`**: Comparing different model capabilities

## 🏗️ Advanced Configuration

### Custom Memory Configuration

```python
config = CAIMConfig(
    max_memory_size=20000,           # Maximum memories to store
    consolidation_threshold=0.8,     # Similarity threshold for consolidation
    retrieval_top_k=10,             # Number of memories to retrieve
    memory_decay_factor=0.95,       # Memory importance decay rate
    embedding_model="custom-model"   # Custom embedding model
)
```

### Agent Customization

```python
agent = CAIMAgent(name="CustomAgent", config=config, model=model, caim_framework=framework)

# Configure memory behavior
agent.auto_consolidate = True
agent.consolidation_interval = 15
agent.max_context_memories = 8
agent.memory_importance_threshold = 0.7

await agent.initialize()
```

### Storage Backends

CAIM supports multiple storage options:

- **In-Memory**: Fast, temporary storage (default)
- **ChromaDB**: Persistent vector storage
- **PostgreSQL**: Relational memory metadata
- **Redis**: Caching and session management

## 📈 Performance Optimization

### Memory Management

```python
# Optimize memory usage
await caim_framework.cleanup_expired_data(max_age_days=30)

# Force consolidation
await caim_framework.consolidate_session_memories(session_id)

# Monitor performance
insights = await caim_framework.get_memory_insights()
```

### Model Optimization

- **Local Models**: Use quantized versions for memory efficiency
- **API Models**: Implement request batching and caching
- **Hybrid Approach**: Combine local and API models based on use case

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/e2e/ -v                     # End-to-end tests

# Run with coverage
pytest tests/ --cov=caim --cov-report=html
```

## 📖 API Reference

### Core Components

- **`CAIMFramework`**: Main orchestration class
- **`CAIMAgent`**: Memory-enhanced agent implementation
- **`MemoryController`**: Memory access decision making
- **`ShortTermMemory`**: Current conversation context
- **`LongTermMemory`**: Persistent memory storage
- **`MemoryRetriever`**: Context-aware memory retrieval

### Model Integration

- **`BaseModel`**: Abstract model interface
- **`HuggingFaceModel`**: Local transformer models
- **`OpenAIModel`**: API-based models
- **`ModelFactory`**: Model creation and management

### Benchmarking

- **`BenchmarkRunner`**: Comprehensive evaluation suite
- **`EvaluationMetrics`**: Performance measurement tools
- **`MemoryBenchmark`**: Memory-specific evaluations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone <repository-url>
cd caim
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Code Quality

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Automated quality checks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research & Citations

CAIM is inspired by cognitive science research on human memory systems. If you use CAIM in your research, please cite:

```bibtex
@software{caim2024,
  title={CAIM: A Cognitive AI Memory Framework for Enhanced Long-Term Agent Interactions},
  author={CAIM Team},
  year={2024},
  url={https://github.com/example/caim}
}
```

## 📞 Support

- **Documentation**: [Full documentation](https://caim.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/example/caim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/caim/discussions)
- **Email**: caim-support@example.com

## 🗺️ Roadmap

### Current Version (v0.1.0)
- ✅ Core memory architecture
- ✅ Multi-model support
- ✅ Basic benchmarking
- ✅ CLI and web interfaces

### Upcoming Features
- 🔄 Advanced memory consolidation algorithms
- 🔄 Multi-modal memory support (text, images, audio)
- 🔄 Distributed memory architectures
- 🔄 Integration with popular AI frameworks
- 🔄 Advanced personalization algorithms
- 🔄 Memory sharing between agents

## 🙏 Acknowledgments

- Inspired by cognitive science research on human memory
- Built on top of excellent open-source projects:
  - HuggingFace Transformers
  - ChromaDB
  - Streamlit
  - FastAPI
  - And many others

---

**CAIM: Bringing human-like memory to AI agents** 🧠✨