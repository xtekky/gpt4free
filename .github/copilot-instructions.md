# gpt4free - Python Library for AI Provider Access

gpt4free is a Python 3.10+ library that provides unified access to 108+ AI providers for text generation, image generation, audio processing, and video generation. It offers a CLI, API server with web GUI, and Python library interface.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Install Dependencies
- Install Python 3.10+ (3.12 recommended): Project requires Python 3.10 or higher
- Clone repository: `git clone https://github.com/xtekky/gpt4free.git && cd gpt4free`
- Install minimal requirements: `pip install -r requirements-min.txt` -- takes 30-60 seconds. NEVER CANCEL.
- Install full requirements: `pip install -r requirements.txt` -- takes 2-5 minutes. NEVER CANCEL. Set timeout to 600+ seconds.
- Remove nodriver (CI requirement): `pip uninstall -y nodriver`
- Install in editable mode: `pip install -e .` -- takes 30 seconds. NEVER CANCEL.

### Testing and Validation
- Run unit tests: `python -m etc.unittest` -- takes 3-5 seconds. NEVER CANCEL.
- Expected results: ~41 tests, 1-2 failures expected (network isolation), 5-8 skipped tests
- Time unit tests: `time python -m etc.unittest` for precise timing
- Individual test scripts available in `etc/testing/` but may have outdated model references

### Running the Application
- **CLI Help**: `g4f --help` or `python -m g4f --help`
- **Client CLI**: `g4f client --help` for interactive text generation
- **API Server**: `python -m g4f --port 8080` -- starts FastAPI server with web GUI on http://localhost:8080
- **API Server Only**: `python -m g4f.cli api --port 8080`
- **Python Library**: 
  ```python
  from g4f.client import Client
  client = Client()
  response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
  print(response.choices[0].message.content)
  ```

## Validation Scenarios

Always test these scenarios after making changes:

### Basic Functionality Test
1. **Import Test**: `python -c "from g4f.client import Client; client = Client(); print('Success')"`
2. **CLI Test**: `g4f --help` should show available commands without errors
3. **Unit Tests**: `python -m etc.unittest` should complete without hanging
4. **API Server Test**: Start server `python -m g4f --port 8080`, then `curl -I http://localhost:8080` should return 200 OK

### Manual Validation Requirements
- ALWAYS test complete user workflows, not just imports
- Test both CLI and Python library interfaces when making provider changes
- Test API server startup and basic HTTP response
- Verify no new import errors or dependency conflicts

## Critical Timing and Build Information

- **NEVER CANCEL** long-running operations. Builds may take several minutes.
- **Requirements Installation**: 
  - Minimal (`requirements-min.txt`): 30-60 seconds. Set timeout to 120+ seconds.
  - Full (`requirements.txt`): 2-5 minutes. Set timeout to 600+ seconds.
- **Unit Tests**: 3-5 seconds. Set timeout to 30+ seconds.
- **Package Installation**: 30-60 seconds. Set timeout to 120+ seconds.
- **Warning**: `pydub` will show ffmpeg warning - this is expected and harmless

## Key Development Info

### Project Structure
- `g4f/` - Main library code
  - `Provider/` - AI provider implementations (108+ providers)
  - `client/` - Client interfaces
  - `api/` - FastAPI server implementation
  - `gui/` - Web GUI components
  - `cli/` - Command line interfaces
- `etc/` - Testing and development tools
  - `unittest/` - Unit test suite
  - `testing/` - Integration tests (may have outdated references)
  - `examples/` - Usage examples
- `docs/` - Documentation
- `docker/` - Docker configuration files

### Provider System
- 108+ AI providers supported including OpenAI, Anthropic, Google, Meta, etc.
- Each provider in `g4f/Provider/` directory
- Working providers listed in CLI help: `g4f --help`
- Test provider availability: Network-dependent, many may fail in isolated environments

### Common Commands Reference
```bash
# Development workflow
git clone https://github.com/xtekky/gpt4free.git
cd gpt4free
pip install -r requirements-min.txt  # Basic deps
pip install -r requirements.txt      # Full deps  
pip uninstall -y nodriver           # CI requirement
pip install -e .                    # Editable install
python -m etc.unittest              # Run tests

# Usage examples
g4f client "Hello world"            # CLI chat
python -m g4f --port 8080           # Start server
python etc/examples/messages.py     # Example script
```

### Installation Alternatives
- **PyPI**: `pip install -U g4f[all]` (external, not for development)
- **Docker**: Use provided docker-compose.yml for containerized deployment
- **Partial installs**: See https://github.com/gpt4free/g4f.dev/tree/main/docs for component-specific installs

### Known Issues and Workarounds
- **ffmpeg warning**: Expected from pydub library, does not affect functionality
- **Network timeouts**: Expected for many providers in isolated environments
- **Model availability**: Provider-dependent, some models may not be accessible
- **nodriver removal**: Required for CI compatibility, remove after full install

## Development Guidelines

### Making Changes
- Always run unit tests before and after changes: `python -m etc.unittest`
- Test both minimal and full installations when adding dependencies
- Validate CLI commands work: `g4f --help`, `g4f client --help`
- Test Python import: `python -c "import g4f; print('OK')"`
- For provider changes, test basic client creation and simple completions

### Before Committing
- Run full test suite: `python -m etc.unittest` 
- Verify no new dependency conflicts
- Test installation from scratch in clean environment when possible
- Ensure backwards compatibility with Python 3.10+

### Timeout Guidelines
- Use 600+ second timeouts for `pip install -r requirements.txt`
- Use 120+ second timeouts for basic pip installations  
- Use 30+ second timeouts for unit tests
- NEVER use default timeouts for package installations - they will fail

### Entry Points
- **CLI Binary**: `g4f` (installed via setuptools entry_points)
- **Module**: `python -m g4f` 
- **Client Module**: `python -m g4f.client`
- **Python Import**: `import g4f` or `from g4f.client import Client`