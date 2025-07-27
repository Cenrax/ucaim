"""Command-line interface for CAIM framework."""

import click
import asyncio
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .core.caim_framework import CAIMFramework
from .core.config import CAIMConfig
from .agents.caim_agent import CAIMAgent
from .models.model_factory import create_model, get_model_recommendations
from .benchmarks.benchmark_runner import BenchmarkRunner
from .demo.streamlit_app import main as run_streamlit_demo


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """CAIM: Cognitive AI Memory Framework CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        caim_config = CAIMConfig.load(config)
    else:
        caim_config = CAIMConfig.from_env()
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = caim_config


@cli.command()
@click.option('--model', '-m', default='Qwen/Qwen2.5-1.5B-Instruct', help='Model to use for the agent')
@click.option('--session-id', '-s', help='Session ID for the conversation')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive chat mode')
@click.argument('message', required=False)
@click.pass_context
def chat(ctx, model, session_id, interactive, message):
    """Chat with a CAIM agent."""
    
    async def run_chat():
        config = ctx.obj['config']
        
        # Create CAIM framework
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        
        # Create model and agent
        model_instance = create_model(model, config)
        agent = CAIMAgent(
            name="CLI-Agent",
            config=config,
            model=model_instance,
            caim_framework=caim_framework
        )
        
        await agent.initialize()
        
        session_id_actual = session_id or f"cli_session_{int(asyncio.get_event_loop().time())}"
        
        if interactive:
            click.echo("🧠 CAIM Interactive Chat Mode (type 'quit' to exit)")
            click.echo(f"Session ID: {session_id_actual}")
            click.echo(f"Model: {model}")
            click.echo("-" * 50)
            
            while True:
                try:
                    user_input = click.prompt("You", type=str)
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    click.echo("🤔 Thinking...")
                    response = await agent.process_message(user_input, session_id_actual)
                    
                    click.echo(f"Assistant: {response.content}")
                    
                    if response.relevant_memories:
                        click.echo(f"💭 Used {len(response.relevant_memories)} memories")
                    
                    click.echo("-" * 50)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
        
        elif message:
            response = await agent.process_message(message, session_id_actual)
            click.echo(response.content)
        
        else:
            click.echo("Please provide a message or use --interactive mode")
        
        # Cleanup
        await agent.shutdown()
        await caim_framework.shutdown()
    
    asyncio.run(run_chat())


@cli.command()
@click.option('--agents', '-a', multiple=True, help='Agents to benchmark (can specify multiple)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--full', '-f', is_flag=True, help='Run full benchmark suite')
@click.option('--memory-only', '-m', is_flag=True, help='Run only memory benchmarks')
@click.pass_context
def benchmark(ctx, agents, output, full, memory_only):
    """Run benchmarks on CAIM framework."""
    
    async def run_benchmark():
        config = ctx.obj['config']
        
        # Default models if none specified
        if not agents:
            agent_models = ['Qwen/Qwen2.5-1.5B-Instruct', 'google/gemma-2b-it']
        else:
            agent_models = list(agents)
        
        # Create CAIM framework
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        
        # Create agents
        agent_list = []
        for model_name in agent_models:
            try:
                model_instance = create_model(model_name, config)
                agent = CAIMAgent(
                    name=f"Benchmark-{model_name.split('/')[-1]}",
                    config=config,
                    model=model_instance,
                    caim_framework=caim_framework
                )
                await agent.initialize()
                agent_list.append(agent)
                click.echo(f"✅ Created agent with {model_name}")
                
            except Exception as e:
                click.echo(f"❌ Failed to create agent with {model_name}: {e}")
        
        if not agent_list:
            click.echo("No agents available for benchmarking")
            return
        
        # Run benchmarks
        runner = BenchmarkRunner(config)
        
        if memory_only:
            click.echo("🧠 Running memory benchmarks...")
            results = await runner.run_memory_benchmark_suite(caim_framework)
        elif full:
            click.echo("🚀 Running full benchmark suite...")
            results = await runner.run_full_benchmark_suite(agent_list, caim_framework)
        else:
            click.echo("📊 Running agent comparison benchmark...")
            results = await runner.run_agent_comparison_benchmark(agent_list)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("BENCHMARK RESULTS")
        click.echo("="*50)
        
        if isinstance(results, dict):
            if 'error' in results:
                click.echo(f"❌ Benchmark failed: {results['error']}")
            else:
                # Display key metrics
                if 'overall_score' in results:
                    click.echo(f"Overall Score: {results['overall_score']:.3f}")
                
                if 'comparison_results' in results:
                    click.echo("\nAgent Performance Comparison:")
                    for agent_name, agent_results in results['comparison_results'].items():
                        metrics = agent_results.get('performance_metrics', {})
                        avg_time = metrics.get('avg_response_time', 0)
                        success_rate = metrics.get('success_rate', 0)
                        click.echo(f"  {agent_name}: {avg_time:.2f}s avg, {success_rate*100:.1f}% success")
                
                if 'summary' in results and 'overall_winner' in results['summary']:
                    winner = results['summary']['overall_winner']
                    click.echo(f"\n🏆 Best Performer: {winner}")
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to {output}")
        
        # Cleanup
        for agent in agent_list:
            await agent.shutdown()
        await caim_framework.shutdown()
    
    asyncio.run(run_benchmark())


@cli.command()
@click.option('--use-case', '-u', default='general', help='Use case for recommendations')
@click.option('--max-memory', '-m', type=int, default=8, help='Maximum memory in GB')
@click.option('--local-only', '-l', is_flag=True, help='Only recommend local models')
def recommend(use_case, max_memory, local_only):
    """Get model recommendations based on requirements."""
    
    constraints = {
        "max_memory_gb": max_memory,
        "require_local": local_only,
        "max_cost_per_1k_tokens": 0.01
    }
    
    recommendations = get_model_recommendations(use_case, constraints)
    
    click.echo(f"🤖 Model Recommendations for '{use_case}' use case:")
    click.echo(f"Constraints: Max {max_memory}GB RAM, {'Local only' if local_only else 'Local/API'}")
    click.echo("-" * 60)
    
    if not recommendations:
        click.echo("❌ No models match your requirements")
        return
    
    for i, rec in enumerate(recommendations, 1):
        click.echo(f"{i}. {rec['model_name']} ({rec['provider']})")
        click.echo(f"   Memory: {rec['estimated_memory_gb']}GB")
        click.echo(f"   Cost: ${rec['estimated_cost_per_1k']} per 1K tokens")
        click.echo(f"   Pros: {', '.join(rec['pros'])}")
        click.echo(f"   Cons: {', '.join(rec['cons'])}")
        click.echo()


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output directory for config file')
def init(output):
    """Initialize a new CAIM configuration."""
    
    config_dir = Path(output) if output else Path.cwd()
    config_file = config_dir / "caim_config.json"
    
    # Create default configuration
    config = CAIMConfig.from_env()
    
    # Interactive configuration
    click.echo("🔧 CAIM Configuration Setup")
    click.echo("-" * 30)
    
    # Model selection
    model_options = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "google/gemma-2b-it",
        "gpt-4o-mini"
    ]
    
    click.echo("Available models:")
    for i, model in enumerate(model_options, 1):
        click.echo(f"  {i}. {model}")
    
    model_choice = click.prompt("Select default model (1-4)", type=int, default=1)
    if 1 <= model_choice <= len(model_options):
        default_model = model_options[model_choice - 1]
        click.echo(f"Selected: {default_model}")
    
    # Memory settings
    max_memory = click.prompt("Maximum memory size", type=int, default=10000)
    config.max_memory_size = max_memory
    
    retrieval_k = click.prompt("Default retrieval top-k", type=int, default=5)
    config.retrieval_top_k = retrieval_k
    
    # API keys
    if click.confirm("Configure OpenAI API key?"):
        api_key = click.prompt("OpenAI API key", hide_input=True)
        config.openai_api_key = api_key
    
    if click.confirm("Configure HuggingFace token?"):
        token = click.prompt("HuggingFace token", hide_input=True)
        config.huggingface_api_token = token
    
    # Save configuration
    config.save(str(config_file))
    
    click.echo(f"✅ Configuration saved to {config_file}")
    click.echo(f"📝 You can edit {config_file} or set environment variables")


@cli.command()
@click.option('--port', '-p', type=int, default=8501, help='Port for Streamlit app')
@click.option('--host', '-h', default='localhost', help='Host for Streamlit app')
def demo(port, host):
    """Launch the Streamlit demo application."""
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Get the path to the streamlit app
        app_path = Path(__file__).parent / "demo" / "streamlit_app.py"
        
        # Modify sys.argv to pass arguments to streamlit
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.port", str(port),
            "--server.address", host
        ]
        
        # Run streamlit
        stcli.main()
        
    except ImportError:
        click.echo("❌ Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        click.echo(f"❌ Error launching demo: {e}")


@cli.command()
@click.option('--session-id', '-s', help='Session ID to inspect')
@click.option('--export', '-e', type=click.Path(), help='Export memories to file')
@click.pass_context
def memory(ctx, session_id, export):
    """Inspect and manage memory system."""
    
    async def run_memory_inspector():
        config = ctx.obj['config']
        
        # Create CAIM framework
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        
        # Get memory insights
        insights = await caim_framework.get_memory_insights(session_id)
        
        click.echo("🧠 CAIM Memory System Status")
        click.echo("=" * 40)
        
        # Display general statistics
        stm_stats = insights.get("short_term_memory", {})
        ltm_stats = insights.get("long_term_memory", {})
        
        click.echo(f"Short-term memories: {stm_stats.get('total_memories', 0)}")
        click.echo(f"Long-term memories: {ltm_stats.get('total_memories', 0)}")
        click.echo(f"Inductive thoughts: {ltm_stats.get('inductive_thoughts', 0)}")
        click.echo(f"Active sessions: {insights.get('active_sessions', 0)}")
        
        if session_id:
            session_insights = insights.get("session_insights", {})
            if session_insights:
                click.echo(f"\nSession '{session_id}' details:")
                click.echo(f"  Total memories: {session_insights.get('total_memories', 0)}")
                click.echo(f"  Interactions: {session_insights.get('total_interactions', 0)}")
                click.echo(f"  Average importance: {session_insights.get('average_importance', 0):.2f}")
        
        # Memory type distribution
        type_dist = ltm_stats.get("type_distribution", {})
        if type_dist:
            click.echo("\nMemory type distribution:")
            for mem_type, count in type_dist.items():
                click.echo(f"  {mem_type}: {count}")
        
        # Export if requested
        if export:
            with open(export, 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            click.echo(f"\n💾 Memory data exported to {export}")
        
        await caim_framework.shutdown()
    
    asyncio.run(run_memory_inspector())


@cli.command()
@click.option('--max-age', '-a', type=int, default=30, help='Maximum age in days for cleanup')
@click.pass_context
def cleanup(ctx, max_age):
    """Clean up expired memories and optimize storage."""
    
    async def run_cleanup():
        config = ctx.obj['config']
        
        # Create CAIM framework
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        
        click.echo(f"🧹 Cleaning up memories older than {max_age} days...")
        
        # Run cleanup
        results = await caim_framework.cleanup_expired_data(max_age)
        
        click.echo("Cleanup results:")
        click.echo(f"  STM memories cleaned: {results.get('stm_memories_cleaned', 0)}")
        click.echo(f"  LTM memories cleaned: {results.get('ltm_memories_cleaned', 0)}")
        click.echo(f"  Store memories cleaned: {results.get('store_memories_cleaned', 0)}")
        
        click.echo("✅ Cleanup completed")
        
        await caim_framework.shutdown()
    
    asyncio.run(run_cleanup())


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()