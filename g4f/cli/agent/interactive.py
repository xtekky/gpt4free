"""Interactive mode for G4F CLI Agent."""

from rich.console import Console
from rich.panel import Panel

console = Console()

def interactive_mode(agent):
    console.print("\n[bold cyan]🤖 G4F CLI Agent[/bold cyan]")
    console.print("Type 'help' for commands, 'exit' to quit")
    
    while True:
        try:
            prompt = console.input("[bold green]> [/bold green]").strip()
            if not prompt: continue

            cmd_map = { "exit": "quit", "quit": "quit", "q": "quit", "help": "help", "clear": "clear", "undo": "undo", "ls": "list" }
            command = cmd_map.get(prompt.lower().split()[0])
            
            if command == "quit": break
            elif command == "help": console.print(Panel("[bold]Commands:[/bold] ls, read <file>, edit <file>, create <file>, undo, help, exit", border_style="cyan"))
            elif command == "clear": console.clear()
            elif command == "undo": agent.undo_last_action()
            elif command == "list": agent.run(f"list directory {prompt.split()[1] if len(prompt.split()) > 1 else '.'}")
            else: agent.run(prompt)
                
        except KeyboardInterrupt: console.print("\nUse 'exit' or 'q' to quit.")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if agent.debug_mode: import traceback; traceback.print_exc()
