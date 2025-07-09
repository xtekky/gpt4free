"""Testing utilities."""

import subprocess
import json
from pathlib import Path
from typing import List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()


class TestRunner:
    """Run and manage tests."""
    
    def __init__(self):
        self.console = console
    
    def detect_test_framework(self) -> Optional[str]:
        """Detect which test framework is being used."""
        # Python
        if any(Path(".").rglob("test_*.py")) or any(Path(".").rglob("*_test.py")):
            if Path("pytest.ini").exists() or Path("setup.cfg").exists():
                return "pytest"
            return "unittest"
        
        # JavaScript/TypeScript
        if Path("package.json").exists():
            try:
                with open("package.json", "r") as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    
                    if "test" in scripts:
                        test_cmd = scripts["test"]
                        if "jest" in test_cmd:
                            return "jest"
                        elif "mocha" in test_cmd:
                            return "mocha"
                        elif "vitest" in test_cmd:
                            return "vitest"
                        elif "ava" in test_cmd:
                            return "ava"
            except:
                pass
        
        # Go
        if any(Path(".").rglob("*_test.go")):
            return "go"
        
        # Rust
        if Path("Cargo.toml").exists() and any(Path(".").rglob("**/tests/*.rs")):
            return "cargo"
        
        # Ruby
        if any(Path(".").rglob("*_spec.rb")) or any(Path(".").rglob("test_*.rb")):
            if Path("Gemfile").exists():
                with open("Gemfile", "r") as f:
                    content = f.read()
                    if "rspec" in content:
                        return "rspec"
            return "minitest"
        
        return None
    
    def run_tests(self, pattern: str = None) -> None:
        """Run tests based on detected framework."""
        framework = self.detect_test_framework()
        
        if not framework:
            self.console.print("[yellow]No test framework detected[/yellow]")
            self._suggest_test_setup()
            return
        
        self.console.print(f"[cyan]Running tests with {framework}...[/cyan]\n")
        
        if framework == "pytest":
            self._run_pytest(pattern)
        elif framework == "unittest":
            self._run_unittest(pattern)
        elif framework == "jest":
            self._run_jest(pattern)
        elif framework == "mocha":
            self._run_mocha(pattern)
        elif framework == "go":
            self._run_go_test(pattern)
        elif framework == "cargo":
            self._run_cargo_test(pattern)
        elif framework == "rspec":
            self._run_rspec(pattern)
        else:
            self.console.print(f"[yellow]Test framework {framework} not fully supported yet[/yellow]")
    
    def _run_pytest(self, pattern: str = None) -> None:
        """Run pytest tests."""
        cmd = ["pytest", "-v", "--tb=short"]
        
        if pattern:
            cmd.append(f"-k={pattern}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            # Parse output for better display
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]pytest not found. Install with: pip install pytest[/red]")
    
    def _run_unittest(self, pattern: str = None) -> None:
        """Run unittest tests."""
        cmd = ["python", "-m", "unittest", "discover", "-v"]
        
        if pattern:
            cmd.extend(["-p", f"*{pattern}*.py"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error running tests: {e}[/red]")
    
    def _run_jest(self, pattern: str = None) -> None:
        """Run Jest tests."""
        cmd = ["npx", "jest", "--colors"]
        
        if pattern:
            cmd.append(pattern)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]Jest not found. Install with: npm install --save-dev jest[/red]")
    
    def _run_mocha(self, pattern: str = None) -> None:
        """Run Mocha tests."""
        cmd = ["npx", "mocha"]
        
        if pattern:
            cmd.extend(["--grep", pattern])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]Mocha not found. Install with: npm install --save-dev mocha[/red]")
    
    def _run_go_test(self, pattern: str = None) -> None:
        """Run Go tests."""
        cmd = ["go", "test", "-v", "./..."]
        
        if pattern:
            cmd.extend(["-run", pattern])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]Go not found. Install from https://golang.org[/red]")
    
    def _run_cargo_test(self, pattern: str = None) -> None:
        """Run Rust tests."""
        cmd = ["cargo", "test"]
        
        if pattern:
            cmd.append(pattern)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]Cargo not found. Install Rust from https://rustup.rs[/red]")
    
    def _run_rspec(self, pattern: str = None) -> None:
        """Run RSpec tests."""
        cmd = ["bundle", "exec", "rspec", "--color"]
        
        if pattern:
            cmd.append(f"--pattern=*{pattern}*")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ All tests passed![/green]")
            else:
                self.console.print("[red]✗ Some tests failed[/red]")
            
            # Show output
            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
                
        except FileNotFoundError:
            self.console.print("[red]RSpec not found. Add to Gemfile: gem 'rspec'[/red]")
    
    def _suggest_test_setup(self) -> None:
        """Suggest test setup based on project type."""
        self.console.print("\n[yellow]No tests found. Here's how to set up testing:[/yellow]\n")
        
        # Check for Python project
        if any(Path(".").rglob("*.py")):
            self.console.print("[cyan]For Python:[/cyan]")
            self.console.print("1. Install pytest: pip install pytest")
            self.console.print("2. Create test files: test_*.py or *_test.py")
            self.console.print("3. Write tests using pytest or unittest")
            self.console.print()
        
        # Check for JavaScript project
        if Path("package.json").exists():
            self.console.print("[cyan]For JavaScript:[/cyan]")
            self.console.print("1. Install Jest: npm install --save-dev jest")
            self.console.print("2. Add to package.json scripts: \"test\": \"jest\"")
            self.console.print("3. Create test files: *.test.js or *.spec.js")
            self.console.print()
        
        # Check for Go project
        if any(Path(".").rglob("*.go")):
            self.console.print("[cyan]For Go:[/cyan]")
            self.console.print("1. Create test files: *_test.go")
            self.console.print("2. Write tests using testing package")
            self.console.print("3. Run with: go test")
            self.console.print()
    
    def generate_test_file(self, source_file: str) -> None:
        """Generate a test file for given source file."""
        source_path = Path(source_file)
        
        if not source_path.exists():
            self.console.print(f"[red]Source file not found: {source_file}[/red]")
            return
        
        # Determine test file name and framework
        if source_path.suffix == ".py":
            test_file = source_path.parent / f"test_{source_path.name}"
            self._generate_python_test(source_path, test_file)
        elif source_path.suffix in [".js", ".ts"]:
            test_file = source_path.parent / f"{source_path.stem}.test{source_path.suffix}"
            self._generate_javascript_test(source_path, test_file)
        elif source_path.suffix == ".go":
            test_file = source_path.parent / f"{source_path.stem}_test.go"
            self._generate_go_test(source_path, test_file)
        else:
            self.console.print(f"[yellow]Test generation not supported for {source_path.suffix} files[/yellow]")
    
    def _generate_python_test(self, source_path: Path, test_path: Path) -> None:
        """Generate Python test file."""
        if test_path.exists():
            self.console.print(f"[yellow]Test file already exists: {test_path}[/yellow]")
            return
        
        # Read source file to find functions/classes
        source_content = source_path.read_text()
        
        # Basic test template
        test_content = f'''"""Tests for {source_path.name}"""

import pytest
from {source_path.stem} import *


class Test{source_path.stem.title()}:
    """Test cases for {source_path.stem}"""
    
    def setup_method(self):
        """Setup for each test method"""
        pass
    
    def teardown_method(self):
        """Teardown for each test method"""
        pass
    
    def test_example(self):
        """Example test case"""
        # TODO: Implement actual tests
        assert True
    
    # Add more test methods here


if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        test_path.write_text(test_content)
        self.console.print(f"[green]✓ Created test file: {test_path}[/green]")
        
        # Show preview
        syntax = Syntax(test_content, "python", theme="monokai", line_numbers=True)
        self.console.print("\n[cyan]Test file preview:[/cyan]")
        self.console.print(syntax)
    
    def _generate_javascript_test(self, source_path: Path, test_path: Path) -> None:
        """Generate JavaScript test file."""
        if test_path.exists():
            self.console.print(f"[yellow]Test file already exists: {test_path}[/yellow]")
            return
        
        # Basic test template
        test_content = f'''// Tests for {source_path.name}

const {{ /* import functions/classes */ }} = require('./{source_path.stem}');

describe('{source_path.stem}', () => {{
  beforeEach(() => {{
    // Setup before each test
  }});
  
  afterEach(() => {{
    // Cleanup after each test
  }});
  
  test('example test', () => {{
    // TODO: Implement actual tests
    expect(true).toBe(true);
  }});
  
  // Add more tests here
}});
'''
        
        test_path.write_text(test_content)
        self.console.print(f"[green]✓ Created test file: {test_path}[/green]")
        
        # Show preview
        syntax = Syntax(test_content, "javascript", theme="monokai", line_numbers=True)
        self.console.print("\n[cyan]Test file preview:[/cyan]")
        self.console.print(syntax)
    
    def _generate_go_test(self, source_path: Path, test_path: Path) -> None:
        """Generate Go test file."""
        if test_path.exists():
            self.console.print(f"[yellow]Test file already exists: {test_path}[/yellow]")
            return
        
        # Read package name from source
        source_content = source_path.read_text()
        package_match = source_content.split('\n')[0]
        package_name = "main"
        if package_match.startswith("package "):
            package_name = package_match.replace("package ", "").strip()
        
        # Basic test template
        test_content = f'''// Tests for {source_path.name}

package {package_name}

import (
    "testing"
)

func TestExample(t *testing.T) {{
    // TODO: Implement actual tests
    got := true
    want := true
    
    if got != want {{
        t.Errorf("got %v, want %v", got, want)
    }}
}}

// Add more test functions here
'''
        
        test_path.write_text(test_content)
        self.console.print(f"[green]✓ Created test file: {test_path}[/green]")
        
        # Show preview
        syntax = Syntax(test_content, "go", theme="monokai", line_numbers=True)
        self.console.print("\n[cyan]Test file preview:[/cyan]")
        self.console.print(syntax)
    
    def coverage_report(self) -> None:
        """Generate test coverage report."""
        framework = self.detect_test_framework()
        
        if not framework:
            self.console.print("[yellow]No test framework detected[/yellow]")
            return
        
        self.console.print(f"[cyan]Generating coverage report with {framework}...[/cyan]\n")
        
        try:
            if framework == "pytest":
                # Install coverage if needed
                subprocess.run(["pip", "install", "pytest-cov"], capture_output=True)
                
                result = subprocess.run(
                    ["pytest", "--cov=.", "--cov-report=term-missing"],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    self.console.print(result.stdout)
                    
            elif framework == "jest":
                result = subprocess.run(
                    ["npx", "jest", "--coverage"],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    self.console.print(result.stdout)
                    
            elif framework == "go":
                result = subprocess.run(
                    ["go", "test", "-cover", "./..."],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    self.console.print(result.stdout)
                    
            else:
                self.console.print(f"[yellow]Coverage not supported for {framework} yet[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Error generating coverage: {e}[/red]")
    
    def watch_tests(self, pattern: str = None) -> None:
        """Run tests in watch mode."""
        framework = self.detect_test_framework()
        
        if not framework:
            self.console.print("[yellow]No test framework detected[/yellow]")
            return
        
        self.console.print(f"[cyan]Starting test watcher with {framework}...[/cyan]")
        self.console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        try:
            if framework == "pytest":
                # Install pytest-watch if needed
                subprocess.run(["pip", "install", "pytest-watch"], capture_output=True)
                
                cmd = ["ptw", "--", "-v"]
                if pattern:
                    cmd.extend(["-k", pattern])
                    
                subprocess.run(cmd)
                
            elif framework == "jest":
                cmd = ["npx", "jest", "--watch"]
                if pattern:
                    cmd.append(pattern)
                    
                subprocess.run(cmd)
                
            else:
                self.console.print(f"[yellow]Watch mode not supported for {framework} yet[/yellow]")
                self.console.print("[dim]You can use a file watcher tool like 'entr' or 'watchman'[/dim]")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Test watcher stopped[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error running test watcher: {e}[/red]")
