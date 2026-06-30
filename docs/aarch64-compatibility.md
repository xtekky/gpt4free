# aarch64 (ARM64) Compatibility

This document describes the compatibility status and known issues for g4f on aarch64 (ARM64) systems.

## Issue Resolution

**Fixed in this release:** The "Illegal instruction (core dumped)" error that occurred when importing g4f on aarch64 systems has been resolved.

### Problem
Previously, g4f would crash with "Illegal instruction (core dumped)" on ARM64 systems (such as Apple Silicon Macs, Raspberry Pi, AWS Graviton instances, etc.) due to compiled dependencies with architecture-specific optimizations.

### Solution
The library now includes proper error handling for architecture-incompatible dependencies:
- Safe import mechanisms prevent crashes when compiled libraries are unavailable
- Graceful fallbacks to alternative implementations when possible
- Clear error messages when specific features require unavailable dependencies

## Compatibility Status

### ✅ Working Features
- Basic client functionality (`from g4f.client import Client`)
- CLI commands (`g4f --help`, `g4f client --help`)
- Providers that use standard HTTP libraries
- Most text generation functionality

### ⚠️ Limited Features  
Some advanced features may have reduced functionality on aarch64:
- Providers requiring `curl_cffi` will fall back to `aiohttp`
- Browser automation features may not be available
- Some performance optimizations may not be active

### 📋 Requirements
For full functionality on aarch64, ensure you have:
```bash
# Basic requirements (should work on all architectures)
pip install -r requirements-min.txt

# Full requirements (some packages may need compilation on aarch64)
pip install -r requirements.txt
```

## Testing Your Installation

You can verify your installation works correctly:

```python
# Test basic import
from g4f.client import Client
client = Client()
print("✓ g4f imported successfully")

# Test CLI
import subprocess
result = subprocess.run(['g4f', '--help'], capture_output=True)
print("✓ CLI works" if result.returncode == 0 else "✗ CLI issues")
```

## Known Issues

1. **Performance**: Some providers may have reduced performance due to fallback implementations
2. **Browser Features**: nodriver and webview functionality may not be available
3. **Image Processing**: Some image-related features may have compatibility issues

## Getting Help

If you encounter issues on aarch64:
1. First try with minimal requirements: `pip install -r requirements-min.txt`
2. Check if the issue persists with basic functionality
3. Report architecture-specific issues with your system details:
   - Architecture: `uname -m`
   - OS: `uname -a` 
   - Python version: `python --version`