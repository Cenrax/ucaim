#!/usr/bin/env python3
"""
Generate architecture diagram for CAIM framework.
Creates a visual representation of the complete codebase architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create comprehensive CAIM architecture diagram."""
    
    # Create figure with larger size for detailed diagram
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'core': '#2E86AB',           # Blue
        'memory': '#A23B72',         # Purple
        'models': '#F18F01',         # Orange
        'storage': '#C73E1D',        # Red
        'interfaces': '#8FBC8F',     # Green
        'agents': '#4A90E2',         # Light Blue
        'utils': '#9B59B6',          # Violet
        'benchmarks': '#E67E22',     # Dark Orange
        'background': '#F8F9FA'      # Light Gray
    }
    
    # Helper function to create rounded rectangles
    def add_component(x, y, width, height, text, color, text_size=10, text_color='white'):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, 
               ha='center', va='center', fontsize=text_size, 
               color=text_color, fontweight='bold', wrap=True)
    
    # Helper function to add arrows
    def add_arrow(x1, y1, x2, y2, color='black', style='->'):
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle=style, shrinkA=5, shrinkB=5,
                              color=color, linewidth=2)
        ax.add_patch(arrow)
    
    # Title
    ax.text(10, 15.5, 'CAIM: Cognitive AI Memory Framework - Complete Architecture', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    # === LAYER 1: User Interfaces (Top) ===
    add_component(1, 13.5, 3.5, 1.2, 'CLI Interface\n• Interactive Chat\n• Benchmarks\n• Memory Mgmt', colors['interfaces'], 9)
    add_component(5, 13.5, 3.5, 1.2, 'Streamlit Demo\n• Web Chat\n• Memory Insights\n• System Status', colors['interfaces'], 9)
    add_component(9, 13.5, 3.5, 1.2, 'FastAPI Server\n• REST Endpoints\n• WebSocket Chat\n• API Docs', colors['interfaces'], 9)
    add_component(13, 13.5, 3.5, 1.2, 'Jupyter Notebooks\n• Examples\n• Benchmarks\n• Tutorials', colors['interfaces'], 9)
    
    # === LAYER 2: Agents Layer ===
    add_component(4, 11.8, 4, 1.2, 'CAIM Agent\n• Message Processing\n• Memory Integration\n• Response Generation', colors['agents'], 10)
    add_component(8.5, 11.8, 4, 1.2, 'Multi-Agent System\n• Agent Coordination\n• Shared Memory\n• Conversation Mgmt', colors['agents'], 10)
    add_component(13, 11.8, 3, 1.2, 'Agent Factory\n• Agent Creation\n• Configuration\n• Lifecycle Mgmt', colors['agents'], 10)
    
    # === LAYER 3: Core CAIM Framework ===
    add_component(2, 9.5, 16, 1.8, 'CAIM Framework Core\n• Memory Controller (Decision Making) • Memory Retrieval (Contextual & Temporal)\n• Post-Thinking (Consolidation & Pattern Extraction) • Session Management • Async Architecture', colors['core'], 11)
    
    # === LAYER 4: Memory System (Main Components) ===
    # Short-term Memory
    add_component(0.5, 7.5, 3, 1.5, 'Short-Term Memory\n• Session Storage\n• Recent Context\n• Conversation Flow\n• Working Memory', colors['memory'], 9)
    
    # Long-term Memory  
    add_component(4, 7.5, 3, 1.5, 'Long-Term Memory\n• Persistent Storage\n• Knowledge Base\n• Experience Archive\n• Pattern Repository', colors['memory'], 9)
    
    # Memory Types
    add_component(7.5, 7.5, 3, 1.5, 'Memory Types\n• Conversational\n• Semantic\n• Procedural\n• Emotional\n• Factual\n• Inductive', colors['memory'], 9)
    
    # Memory Processing
    add_component(11, 7.5, 3, 1.5, 'Memory Processing\n• Consolidation\n• Decay/Forgetting\n• Importance Scoring\n• Retrieval Ranking', colors['memory'], 9)
    
    # Post-Thinking Module
    add_component(14.5, 7.5, 3, 1.5, 'Post-Thinking\n• Pattern Detection\n• Insight Generation\n• Memory Synthesis\n• Learning Analytics', colors['memory'], 9)
    
    # === LAYER 5: Model Layer ===
    add_component(1, 5.5, 3.5, 1.5, 'HuggingFace Models\n• Qwen Integration\n• Gemma 1B/3B\n• Local Inference\n• Model Management', colors['models'], 9)
    add_component(5, 5.5, 3.5, 1.5, 'OpenAI Integration\n• GPT Models\n• API Management\n• Cost Tracking\n• Benchmarking', colors['models'], 9)
    add_component(9, 5.5, 3.5, 1.5, 'Model Abstraction\n• Unified Interface\n• Response Handling\n• Streaming Support\n• Error Management', colors['models'], 9)
    add_component(13, 5.5, 3.5, 1.5, 'Embedding Models\n• Sentence Transformers\n• Vector Generation\n• Similarity Calculation\n• Semantic Search', colors['models'], 9)
    
    # === LAYER 6: Storage Layer ===
    add_component(1, 3.5, 3, 1.5, 'In-Memory Store\n• Fast Access\n• Session Cache\n• Temporary Storage\n• Development Mode', colors['storage'], 9)
    add_component(4.5, 3.5, 3, 1.5, 'Vector Store\n• ChromaDB\n• Semantic Search\n• Similarity Queries\n• Embedding Storage', colors['storage'], 9)
    add_component(8, 3.5, 3, 1.5, 'Database Store\n• PostgreSQL\n• SQLite Support\n• Persistent Storage\n• Query Optimization', colors['storage'], 9)
    add_component(11.5, 3.5, 3, 1.5, 'File System\n• Configuration Files\n• Logs & Analytics\n• Export/Import\n• Backup Storage', colors['storage'], 9)
    add_component(15, 3.5, 3, 1.5, 'Cloud Storage\n• Scalable Storage\n• Distributed Access\n• Backup & Recovery\n• Multi-tenant', colors['storage'], 9)
    
    # === LAYER 7: Support Systems ===
    add_component(1, 1.5, 3, 1.5, 'Configuration\n• CAIM Config\n• Environment Vars\n• Model Settings\n• System Parameters', colors['utils'], 9)
    add_component(4.5, 1.5, 3, 1.5, 'Logging & Monitoring\n• Structured Logging\n• Performance Metrics\n• Error Tracking\n• System Health', colors['utils'], 9)
    add_component(8, 1.5, 3, 1.5, 'Benchmarking\n• Performance Tests\n• Memory Efficiency\n• Model Comparison\n• Quality Metrics', colors['benchmarks'], 9)
    add_component(11.5, 1.5, 3, 1.5, 'Utilities\n• Data Processing\n• Helper Functions\n• Validation\n• Common Tools', colors['utils'], 9)
    add_component(15, 1.5, 3, 1.5, 'Exception Handling\n• Custom Exceptions\n• Error Recovery\n• Graceful Degradation\n• Fault Tolerance', colors['utils'], 9)
    
    # === Data Flow Arrows ===
    # User interfaces to agents
    add_arrow(2.75, 13.5, 5.5, 13.0, colors['interfaces'])
    add_arrow(6.75, 13.5, 6.5, 13.0, colors['interfaces'])
    add_arrow(10.75, 13.5, 10.5, 13.0, colors['interfaces'])
    add_arrow(14.75, 13.5, 14.5, 13.0, colors['interfaces'])
    
    # Agents to CAIM Core
    add_arrow(6, 11.8, 8, 11.3, colors['agents'])
    add_arrow(10.5, 11.8, 10, 11.3, colors['agents'])
    add_arrow(14.5, 11.8, 12, 11.3, colors['agents'])
    
    # CAIM Core to Memory Components
    add_arrow(4, 9.5, 2, 9.0, colors['core'])
    add_arrow(6, 9.5, 5.5, 9.0, colors['core'])
    add_arrow(10, 9.5, 9, 9.0, colors['core'])
    add_arrow(14, 9.5, 12.5, 9.0, colors['core'])
    add_arrow(16, 9.5, 16, 9.0, colors['core'])
    
    # Memory to Models
    add_arrow(2, 7.5, 2.75, 7.0, colors['memory'])
    add_arrow(5.5, 7.5, 6.75, 7.0, colors['memory'])
    add_arrow(9, 7.5, 10.75, 7.0, colors['memory'])
    add_arrow(12.5, 7.5, 14.75, 7.0, colors['memory'])
    
    # Models to Storage
    add_arrow(2.75, 5.5, 2.5, 5.0, colors['models'])
    add_arrow(6.75, 5.5, 6, 5.0, colors['models'])
    add_arrow(10.75, 5.5, 9.5, 5.0, colors['models'])
    add_arrow(14.75, 5.5, 13, 5.0, colors['models'])
    
    # Storage to Utils
    add_arrow(2.5, 3.5, 2.5, 3.0, colors['storage'])
    add_arrow(6, 3.5, 6, 3.0, colors['storage'])
    add_arrow(9.5, 3.5, 9.5, 3.0, colors['storage'])
    add_arrow(13, 3.5, 13, 3.0, colors['storage'])
    add_arrow(16.5, 3.5, 16.5, 3.0, colors['storage'])
    
    # === Legend ===
    legend_x = 0.5
    legend_y = 0.5
    legend_items = [
        ('Core Framework', colors['core']),
        ('Memory System', colors['memory']),
        ('Model Layer', colors['models']),
        ('Storage Layer', colors['storage']),
        ('User Interfaces', colors['interfaces']),
        ('Agent Layer', colors['agents']),
        ('Utilities', colors['utils']),
        ('Benchmarks', colors['benchmarks'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x = legend_x + (i % 4) * 4.5
        y = legend_y - (i // 4) * 0.3
        add_component(x, y, 1, 0.2, '', color, 8)
        ax.text(x + 1.2, y + 0.1, label, fontsize=9, va='center')
    
    # === Key Features Box ===
    features_text = """Key Features:
• Async Architecture with Error Handling
• Multi-Model Support (HF + OpenAI)
• Flexible Storage Backends
• Memory Consolidation & Decay
• Real-time Memory Insights
• Production-Ready CLI & Web Interface"""
    
    ax.text(17, 12, features_text, fontsize=10, va='top', ha='left',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/subham/Desktop/codes/caim/caim_architecture_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/subham/Desktop/codes/caim/caim_architecture_diagram.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("✅ Architecture diagram saved as:")
    print("   📊 caim_architecture_diagram.png (High-res image)")
    print("   📄 caim_architecture_diagram.pdf (Vector format)")
    
    return fig

if __name__ == "__main__":
    create_architecture_diagram()
    plt.show()