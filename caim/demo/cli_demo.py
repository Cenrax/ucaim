"""Command-line demo for CAIM framework."""

import asyncio
import logging
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.caim_framework import CAIMFramework
from ..core.config import CAIMConfig
from ..agents.caim_agent import CAIMAgent
from ..models.model_factory import create_model


logger = logging.getLogger(__name__)


class CLIDemo:
    """Command-line interface demo for CAIM framework."""
    
    def __init__(self):
        self.console = Console()
        self.caim_framework: Optional[CAIMFramework] = None
        self.agent: Optional[CAIMAgent] = None
        self.session_id = "cli_demo_session"
    
    async def run(self):
        """Run the CLI demo."""
        self.console.print(Panel.fit("🧠 CAIM Framework CLI Demo", style="bold blue"))
        
        try:
            # Initialize system
            await self._initialize_system()
            
            # Interactive loop
            await self._interactive_loop()
            
        except KeyboardInterrupt:
            self.console.print("\n👋 Goodbye!")
        except Exception as e:
            self.console.print(f"❌ Error: {e}", style="red")
        finally:
            await self._cleanup()
    
    async def _initialize_system(self):
        """Initialize CAIM system."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Initialize framework
            task1 = progress.add_task("Initializing CAIM framework...", total=1)
            config = CAIMConfig.from_env()
            self.caim_framework = CAIMFramework(config)
            await self.caim_framework.initialize()
            progress.update(task1, completed=1)
            
            # Create model
            task2 = progress.add_task("Loading AI model...", total=1)
            try:
                model = create_model("Qwen/Qwen2.5-1.5B-Instruct", config)
            except Exception:
                # Fallback to mock model
                from ..models.base_model import BaseModel, ModelResponse
                from datetime import datetime
                
                class MockModel(BaseModel):
                    def __init__(self):
                        super().__init__(config, "MockModel")
                    
                    async def initialize(self):
                        self.is_initialized = True
                    
                    async def generate_response(self, prompt, **kwargs):
                        return ModelResponse(
                            content=f"Mock response to: {prompt[:30]}...",
                            model_name=self.model_name,
                            timestamp=datetime.utcnow(),
                            metadata={"mock": True}
                        )
                    
                    async def generate_streaming_response(self, prompt, **kwargs):
                        response = await self.generate_response(prompt, **kwargs)
                        for word in response.content.split():
                            yield word + " "
                
                model = MockModel()
            
            progress.update(task2, completed=1)
            
            # Create agent
            task3 = progress.add_task("Creating CAIM agent...", total=1)
            self.agent = CAIMAgent(
                name="CLI-Agent",
                config=config,
                model=model,
                caim_framework=self.caim_framework
            )
            await self.agent.initialize()
            progress.update(task3, completed=1)
        
        self.console.print("✅ System initialized successfully!", style="green")
    
    async def _interactive_loop(self):
        """Main interactive loop."""
        self.console.print("\n💬 Chat with CAIM (type 'help' for commands, 'quit' to exit)")
        self.console.print("-" * 60)
        
        while True:
            try:
                user_input = self.console.input("\n[bold cyan]You:[/bold cyan] ")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'stats':
                    await self._show_stats()
                elif user_input.lower() == 'memory':
                    await self._show_memory_info()
                elif user_input.lower() == 'clear':
                    await self._clear_session()
                elif user_input.lower().startswith('consolidate'):
                    await self._force_consolidation()
                elif user_input.strip():
                    await self._process_message(user_input)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}", style="red")
    
    async def _process_message(self, message: str):
        """Process user message and show response."""
        with Progress(
            SpinnerColumn(),
            TextColumn("🤔 Thinking..."),
            console=self.console,
        ) as progress:
            progress.add_task("Processing", total=1)
            
            response = await self.agent.process_message(message, self.session_id)
        
        self.console.print(f"[bold green]Agent:[/bold green] {response.content}")
        
        # Show memory info if memories were used
        if response.relevant_memories:
            self.console.print(
                f"💭 [dim]Used {len(response.relevant_memories)} memories[/dim]"
            )
    
    def _show_help(self):
        """Show help information."""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("help", "Show this help message")
        help_table.add_row("stats", "Show agent and system statistics")
        help_table.add_row("memory", "Show memory system information")
        help_table.add_row("clear", "Clear current session")
        help_table.add_row("consolidate", "Force memory consolidation")
        help_table.add_row("quit/exit/bye", "Exit the demo")
        
        self.console.print(help_table)
    
    async def _show_stats(self):
        """Show system statistics."""
        try:
            stats = await self.agent.get_agent_statistics()
            
            stats_table = Table(title="System Statistics", show_header=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Agent Name", stats.get("agent_name", "N/A"))
            stats_table.add_row("Model", stats.get("model_name", "N/A"))
            stats_table.add_row("Total Sessions", str(stats.get("total_sessions", 0)))
            stats_table.add_row("Active Sessions", str(stats.get("active_sessions", 0)))
            stats_table.add_row("Total Interactions", str(stats.get("total_interactions", 0)))
            
            self.console.print(stats_table)
            
        except Exception as e:
            self.console.print(f"❌ Error getting stats: {e}", style="red")
    
    async def _show_memory_info(self):
        """Show memory system information."""
        try:
            insights = await self.caim_framework.get_memory_insights(self.session_id)
            
            memory_table = Table(title="Memory System Info", show_header=True)
            memory_table.add_column("Component", style="cyan")
            memory_table.add_column("Count", style="white")
            
            stm_stats = insights.get("short_term_memory", {})
            ltm_stats = insights.get("long_term_memory", {})
            
            memory_table.add_row("Short-term Memories", str(stm_stats.get("total_memories", 0)))
            memory_table.add_row("Long-term Memories", str(ltm_stats.get("total_memories", 0)))
            memory_table.add_row("Inductive Thoughts", str(ltm_stats.get("inductive_thoughts", 0)))
            memory_table.add_row("Active Sessions", str(insights.get("active_sessions", 0)))
            
            self.console.print(memory_table)
            
        except Exception as e:
            self.console.print(f"❌ Error getting memory info: {e}", style="red")
    
    async def _clear_session(self):
        """Clear current session."""
        try:
            if hasattr(self.agent, 'clear_conversation_history'):
                await self.agent.clear_conversation_history(self.session_id)
            
            self.console.print("✅ Session cleared!", style="green")
            
        except Exception as e:
            self.console.print(f"❌ Error clearing session: {e}", style="red")
    
    async def _force_consolidation(self):
        """Force memory consolidation."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("🔄 Consolidating memories..."),
                console=self.console,
            ) as progress:
                progress.add_task("Consolidating", total=1)
                
                result = await self.caim_framework.consolidate_session_memories(self.session_id)
            
            self.console.print(
                f"✅ Consolidation complete! "
                f"Promoted: {result.get('promoted_memories', 0)}, "
                f"Thoughts: {result.get('inductive_thoughts', 0)}",
                style="green"
            )
            
        except Exception as e:
            self.console.print(f"❌ Error consolidating: {e}", style="red")
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.agent:
                await self.agent.shutdown()
            if self.caim_framework:
                await self.caim_framework.shutdown()
            
            self.console.print("🧹 Cleanup completed", style="dim")
            
        except Exception as e:
            self.console.print(f"❌ Cleanup error: {e}", style="red")


async def main():
    """Main function to run CLI demo."""
    demo = CLIDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())