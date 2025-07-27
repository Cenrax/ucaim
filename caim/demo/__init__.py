"""Demo applications for CAIM framework."""

from .streamlit_app import StreamlitDemo
from .fastapi_app import FastAPIDemo
from .cli_demo import CLIDemo

__all__ = [
    "StreamlitDemo",
    "FastAPIDemo",
    "CLIDemo",
]