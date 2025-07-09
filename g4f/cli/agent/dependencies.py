"""Dependency management utilities."""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table

console = Console()


class DependencyManager:
    """Manage project dependencies."""
    
    def __init__(self):
        self.console = console
    
    def detect_package_managers(self) -> List[str]:
        """Detect available package managers."""
        managers = []
        
        # Python
        if Path("requirements.txt").exists() or Path("setup.py").exists() or Path("pyproject.toml").exists():
            managers.append("pip")
        if Path("Pipfile").exists():
            managers.append("pipenv")
        if Path("poetry.lock").exists():
            managers.append("poetry")
        
        # JavaScript
        if Path("package.json").exists():
            if Path("yarn.lock").exists():
                managers.append("yarn")
            elif Path("pnpm-lock.yaml").exists():
                managers.append("pnpm")
            else:
                managers.append("npm")
        
        # Ruby
        if Path("Gemfile").exists():
            managers.append("bundler")
        
        # Go
        if Path("go.mod").exists():
            managers.append("go")
        
        # Rust
        if Path("Cargo.toml").exists():
            managers.append("cargo")
        
        return managers
    
    def check_dependencies(self, file_path: str = None) -> None:
        """Check if dependencies are installed."""
        if file_path:
            # Check specific file
            path = Path(file_path)
            if path.name == "requirements.txt":
                self._check_python_requirements(path)
            elif path.name == "package.json":
                self._check_node_packages(path)
            elif path.name == "Gemfile":
                self._check_ruby_gems(path)
            elif path.name == "go.mod":
                self._check_go_modules(path)
            elif path.name == "Cargo.toml":
                self._check_rust_crates(path)
        else:
            # Check all
            self.check_all_dependencies()
    
    def check_all_dependencies(self) -> None:
        """Check all project dependencies."""
        managers = self.detect_package_managers()
        
        if not managers:
            self.console.print("[yellow]No package managers detected[/yellow]")
            return
        
        self.console.print("[cyan]Checking project dependencies...[/cyan]\n")
        
        for manager in managers:
            if manager == "pip":
                if Path("requirements.txt").exists():
                    self._check_python_requirements(Path("requirements.txt"))
            elif manager == "npm":
                if Path("package.json").exists():
                    self._check_node_packages(Path("package.json"))
            elif manager == "poetry":
                self._check_poetry_dependencies()
            elif manager == "pipenv":
                self._check_pipenv_dependencies()
            # Add more as needed
    
    def _check_python_requirements(self, req_file: Path) -> None:
        """Check Python requirements."""
        self.console.print(f"[cyan]Checking Python dependencies ({req_file.name})...[/cyan]")
        
        try:
            # Read requirements
            requirements = req_file.read_text().strip().split('\n')
            requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]
            
            # Get installed packages
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True
            )
            
            installed = {}
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==')
                    installed[name.lower()] = version
            
            # Check each requirement
            table = Table(title="Python Dependencies")
            table.add_column("Package", style="cyan")
            table.add_column("Required", style="yellow")
            table.add_column("Installed", style="green")
            table.add_column("Status", style="white")
            
            missing = []
            for req in requirements:
                if '==' in req:
                    name, version = req.split('==')
                    installed_version = installed.get(name.lower(), "Not installed")
                    status = "✓" if installed_version == version else "✗"
                    if status == "✗":
                        missing.append(req)
                    table.add_row(name, version, installed_version, status)
                else:
                    # Handle other version specifiers
                    name = req.split('<')[0].split('>')[0].split('~')[0].strip()
                    installed_version = installed.get(name.lower(), "Not installed")
                    status = "✓" if installed_version != "Not installed" else "✗"
                    if status == "✗":
                        missing.append(req)
                    table.add_row(name, req, installed_version, status)
            
            self.console.print(table)
            
            if missing:
                self.console.print(f"\n[yellow]Missing packages: {', '.join(missing)}[/yellow]")
                if self.console.input("\nInstall missing packages? (y/n): ").lower() == 'y':
                    self.install_python_packages(missing)
            else:
                self.console.print("\n[green]✓ All Python dependencies are installed[/green]")
                
        except Exception as e:
            self.console.print(f"[red]Error checking Python dependencies: {e}[/red]")
    
    def _check_node_packages(self, package_file: Path) -> None:
        """Check Node.js packages."""
        self.console.print(f"[cyan]Checking Node.js dependencies...[/cyan]")
        
        try:
            # Read package.json
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            deps = {}
            if 'dependencies' in package_data:
                deps.update(package_data['dependencies'])
            if 'devDependencies' in package_data:
                deps.update(package_data['devDependencies'])
            
            # Check if node_modules exists
            if not Path("node_modules").exists():
                self.console.print("[yellow]node_modules not found[/yellow]")
                if self.console.input("\nRun npm install? (y/n): ").lower() == 'y':
                    self.run_npm_install()
            else:
                # Check individual packages
                table = Table(title="Node.js Dependencies")
                table.add_column("Package", style="cyan")
                table.add_column("Required", style="yellow")
                table.add_column("Status", style="white")
                
                missing = []
                for name, version in deps.items():
                    if (Path("node_modules") / name).exists():
                        table.add_row(name, version, "✓")
                    else:
                        table.add_row(name, version, "✗")
                        missing.append(f"{name}@{version}")
                
                self.console.print(table)
                
                if missing:
                    self.console.print(f"\n[yellow]Missing packages: {len(missing)}[/yellow]")
                    if self.console.input("\nInstall missing packages? (y/n): ").lower() == 'y':
                        self.run_npm_install()
                else:
                    self.console.print("\n[green]✓ All Node.js dependencies are installed[/green]")
                    
        except Exception as e:
            self.console.print(f"[red]Error checking Node.js dependencies: {e}[/red]")
    
    def _check_poetry_dependencies(self) -> None:
        """Check Poetry dependencies."""
        self.console.print("[cyan]Checking Poetry dependencies...[/cyan]")
        
        try:
            result = subprocess.run(
                ["poetry", "show"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.console.print("[yellow]Poetry dependencies not installed[/yellow]")
                if self.console.input("\nRun poetry install? (y/n): ").lower() == 'y':
                    self.run_poetry_install()
            else:
                self.console.print("[green]✓ Poetry dependencies are installed[/green]")
                
        except FileNotFoundError:
            self.console.print("[red]Poetry not found. Install with: pip install poetry[/red]")
    
    def _check_pipenv_dependencies(self) -> None:
        """Check Pipenv dependencies."""
        self.console.print("[cyan]Checking Pipenv dependencies...[/cyan]")
        
        try:
            result = subprocess.run(
                ["pipenv", "check"],
                capture_output=True,
                text=True
            )
            
            if "All good!" in result.stdout:
                self.console.print("[green]✓ Pipenv dependencies are installed[/green]")
            else:
                self.console.print("[yellow]Pipenv dependencies have issues[/yellow]")
                if self.console.input("\nRun pipenv install? (y/n): ").lower() == 'y':
                    self.run_pipenv_install()
                    
        except FileNotFoundError:
            self.console.print("[red]Pipenv not found. Install with: pip install pipenv[/red]")
    
    def _check_ruby_gems(self, gemfile: Path) -> None:
        """Check Ruby gems."""
        self.console.print("[cyan]Checking Ruby gems...[/cyan]")
        
        try:
            result = subprocess.run(
                ["bundle", "check"],
                capture_output=True,
                text=True
            )
            
            if "The Gemfile's dependencies are satisfied" in result.stdout:
                self.console.print("[green]✓ All Ruby gems are installed[/green]")
            else:
                self.console.print("[yellow]Missing Ruby gems[/yellow]")
                if self.console.input("\nRun bundle install? (y/n): ").lower() == 'y':
                    self.run_bundle_install()
                    
        except FileNotFoundError:
            self.console.print("[red]Bundler not found. Install with: gem install bundler[/red]")
    
    def _check_go_modules(self, go_mod: Path) -> None:
        """Check Go modules."""
        self.console.print("[cyan]Checking Go modules...[/cyan]")
        
        try:
            result = subprocess.run(
                ["go", "mod", "verify"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All Go modules are verified[/green]")
            else:
                self.console.print("[yellow]Go modules need updating[/yellow]")
                if self.console.input("\nRun go mod download? (y/n): ").lower() == 'y':
                    self.run_go_mod_download()
                    
        except FileNotFoundError:
            self.console.print("[red]Go not found. Install from https://golang.org[/red]")
    
    def _check_rust_crates(self, cargo_toml: Path) -> None:
        """Check Rust crates."""
        self.console.print("[cyan]Checking Rust crates...[/cyan]")
        
        try:
            result = subprocess.run(
                ["cargo", "check"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All Rust crates are available[/green]")
            else:
                self.console.print("[yellow]Rust crates need updating[/yellow]")
                if self.console.input("\nRun cargo build? (y/n): ").lower() == 'y':
                    self.run_cargo_build()
                    
        except FileNotFoundError:
            self.console.print("[red]Cargo not found. Install Rust from https://rustup.rs[/red]")
    
    def install_python_packages(self, packages: List[str]) -> None:
        """Install Python packages."""
        self.console.print(f"\n[cyan]Installing Python packages...[/cyan]")
        
        for package in packages:
            try:
                result = subprocess.run(
                    ["pip", "install", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.console.print(f"[green]✓ Installed {package}[/green]")
                else:
                    self.console.print(f"[red]✗ Failed to install {package}[/red]")
                    self.console.print(f"[dim]{result.stderr}[/dim]")
                    
            except Exception as e:
                self.console.print(f"[red]Error installing {package}: {e}[/red]")
    
    def run_npm_install(self) -> None:
        """Run npm install."""
        self.console.print("\n[cyan]Running npm install...[/cyan]")
        
        try:
            result = subprocess.run(
                ["npm", "install"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ npm install completed[/green]")
            else:
                self.console.print("[red]✗ npm install failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running npm install: {e}[/red]")
    
    def run_poetry_install(self) -> None:
        """Run poetry install."""
        self.console.print("\n[cyan]Running poetry install...[/cyan]")
        
        try:
            result = subprocess.run(
                ["poetry", "install"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ poetry install completed[/green]")
            else:
                self.console.print("[red]✗ poetry install failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running poetry install: {e}[/red]")
    
    def run_pipenv_install(self) -> None:
        """Run pipenv install."""
        self.console.print("\n[cyan]Running pipenv install...[/cyan]")
        
        try:
            result = subprocess.run(
                ["pipenv", "install"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ pipenv install completed[/green]")
            else:
                self.console.print("[red]✗ pipenv install failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running pipenv install: {e}[/red]")
    
    def run_bundle_install(self) -> None:
        """Run bundle install."""
        self.console.print("\n[cyan]Running bundle install...[/cyan]")
        
        try:
            result = subprocess.run(
                ["bundle", "install"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ bundle install completed[/green]")
            else:
                self.console.print("[red]✗ bundle install failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running bundle install: {e}[/red]")
    
    def run_go_mod_download(self) -> None:
        """Run go mod download."""
        self.console.print("\n[cyan]Running go mod download...[/cyan]")
        
        try:
            result = subprocess.run(
                ["go", "mod", "download"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ go mod download completed[/green]")
            else:
                self.console.print("[red]✗ go mod download failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running go mod download: {e}[/red]")
    
    def run_cargo_build(self) -> None:
        """Run cargo build."""
        self.console.print("\n[cyan]Running cargo build...[/cyan]")
        
        try:
            result = subprocess.run(
                ["cargo", "build"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ cargo build completed[/green]")
            else:
                self.console.print("[red]✗ cargo build failed[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Error running cargo build: {e}[/red]")
    
    def add_package(self, package: str, manager: str = None, dev: bool = False) -> None:
        """Add a package to the project."""
        if not manager:
            managers = self.detect_package_managers()
            if not managers:
                self.console.print("[red]No package manager detected[/red]")
                return
            manager = managers[0]
        
        self.console.print(f"[cyan]Adding {package} with {manager}...[/cyan]")
        
        try:
            if manager == "pip":
                cmd = ["pip", "install", package]
                subprocess.run(cmd)
                # Update requirements.txt
                subprocess.run(["pip", "freeze", ">", "requirements.txt"], shell=True)
                
            elif manager == "npm":
                cmd = ["npm", "install", package]
                if dev:
                    cmd.append("--save-dev")
                subprocess.run(cmd)
                
            elif manager == "yarn":
                cmd = ["yarn", "add", package]
                if dev:
                    cmd.append("--dev")
                subprocess.run(cmd)
                
            elif manager == "poetry":
                cmd = ["poetry", "add", package]
                if dev:
                    cmd.append("--dev")
                subprocess.run(cmd)
                
            elif manager == "pipenv":
                cmd = ["pipenv", "install", package]
                if dev:
                    cmd.append("--dev")
                subprocess.run(cmd)
                
            self.console.print(f"[green]✓ Added {package}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error adding package: {e}[/red]")
    
    def remove_package(self, package: str, manager: str = None) -> None:
        """Remove a package from the project."""
        if not manager:
            managers = self.detect_package_managers()
            if not managers:
                self.console.print("[red]No package manager detected[/red]")
                return
            manager = managers[0]
        
        self.console.print(f"[cyan]Removing {package} with {manager}...[/cyan]")
        
        try:
            if manager == "pip":
                cmd = ["pip", "uninstall", "-y", package]
                subprocess.run(cmd)
                # Update requirements.txt
                subprocess.run(["pip", "freeze", ">", "requirements.txt"], shell=True)
                
            elif manager == "npm":
                cmd = ["npm", "uninstall", package]
                subprocess.run(cmd)
                
            elif manager == "yarn":
                cmd = ["yarn", "remove", package]
                subprocess.run(cmd)
                
            elif manager == "poetry":
                cmd = ["poetry", "remove", package]
                subprocess.run(cmd)
                
            elif manager == "pipenv":
                cmd = ["pipenv", "uninstall", package]
                subprocess.run(cmd)
                
            self.console.print(f"[green]✓ Removed {package}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error removing package: {e}[/red]")
    
    def update_packages(self, manager: str = None) -> None:
        """Update all packages."""
        if not manager:
            managers = self.detect_package_managers()
            if not managers:
                self.console.print("[red]No package manager detected[/red]")
                return
            manager = managers[0]
        
        self.console.print(f"[cyan]Updating packages with {manager}...[/cyan]")
        
        try:
            if manager == "pip":
                # Update pip itself first
                subprocess.run(["pip", "install", "--upgrade", "pip"])
                # Update all packages
                result = subprocess.run(
                    ["pip", "list", "--outdated", "--format=json"],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    outdated = json.loads(result.stdout)
                    for pkg in outdated:
                        subprocess.run(["pip", "install", "--upgrade", pkg["name"]])
                
            elif manager == "npm":
                subprocess.run(["npm", "update"])
                
            elif manager == "yarn":
                subprocess.run(["yarn", "upgrade"])
                
            elif manager == "poetry":
                subprocess.run(["poetry", "update"])
                
            elif manager == "pipenv":
                subprocess.run(["pipenv", "update"])
                
            self.console.print(f"[green]✓ Packages updated[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error updating packages: {e}[/red]")
