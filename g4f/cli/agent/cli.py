"""Command line interface for G4F Agent."""

import sys
import argparse
import traceback
from .config import load_config
from .agent import EnhancedAgent
from .interactive import interactive_mode

def get_agent_parser():
    """Creates the parser for the agent subcommand."""
    parser = argparse.ArgumentParser(description="G4F Agent - AI-powered coding assistant", add_help=False)
    parser.add_argument("prompt", nargs="?", help="Initial prompt to run.")
    
    # --- MODIFICATION START ---
    # Add the --model argument to match the client's functionality.
    parser.add_argument(
        '-m', '--model',
        help="Model to use (overrides the agent's config.yaml)"
    )
    # --- MODIFICATION END ---
    
    return parser

def run_agent_args(args):
    """The main entry point for the agent mode."""
    # Load the default configuration from the agent's config file first.
    config = load_config()

    # Check if the debug flag was passed from the main parser and apply it.
    if hasattr(args, 'debug') and args.debug:
        config.debug_mode = True

    # --- MODIFICATION START ---
    # If a model was specified on the command line, it overrides the config file.
    if args.model:
        config.model = args.model
    # --- MODIFICATION END ---

    try:
        agent = EnhancedAgent(config)
        if args.prompt:
            agent.run(args.prompt)
        else:
            interactive_mode(agent)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        if config.debug_mode:
            traceback.print_exc()
        else:
            print(f"An unexpected error occurred: {e}")
        sys.exit(1)