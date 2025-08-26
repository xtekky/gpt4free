# Build Workflow Documentation

This document explains the comprehensive build workflow for g4f that creates packages for multiple platforms and package managers.

## Workflow Overview

The `.github/workflows/build-packages.yml` workflow automatically builds multiple package formats when a version tag is pushed to the repository.

### Supported Package Formats

1. **PyPI Package** - Python wheel and source distribution
2. **Windows Executable** - Standalone .exe file built with Nuitka  
3. **Linux Executable** - Standalone binary for Linux systems built with Nuitka
4. **macOS Executable** - Standalone binary for macOS systems built with Nuitka (x64 and ARM64)
5. **Debian Packages** - .deb files for Ubuntu/Debian (amd64, arm64, armhf)
6. **WinGet Package** - Windows Package Manager manifest
7. **Docker Images** - Multi-architecture container images

### Triggering a Build

To trigger a build, push a version tag to the repository:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The workflow will:
1. Detect the tag and extract the version
2. Build all package formats in parallel 
3. Create a GitHub release with all artifacts
4. Publish to PyPI (for releases)
5. Generate WinGet manifest for Windows Package Manager

### Manual Build Triggering

You can also manually trigger builds using the workflow_dispatch event:

1. Go to the "Actions" tab in GitHub
2. Select "Build All Packages" workflow
3. Click "Run workflow"
4. Optionally specify a version number

### Package Locations

After a successful build, packages are available:

- **GitHub Releases**: All executables and packages as release assets
  - Python packages (wheel and source distribution)
  - Standalone executables for Windows, Linux, and macOS
  - Debian packages for AMD64, ARM64, and ARMv7 architectures
  - WinGet manifest files
- **PyPI**: `pip install g4f`
- **Docker Hub**: `docker pull hlohaus789/g4f:latest`
- **WinGet**: `winget install g4f` (after manifest approval)

### Build Requirements

The workflow handles all dependencies automatically, but for local development:

- Python 3.10+
- Nuitka for executables (replaces PyInstaller)
- Docker for container builds
- dpkg-deb for Debian packages

### Customizing Builds

Key files for customization:

- `g4f_cli.py` - Entry point for executable builds
- `scripts/build-nuitka.sh` - Nuitka build script for all platforms
- `scripts/build-deb.sh` - Debian package build script
- `winget/manifests/` - WinGet package manifest templates
- `.github/workflows/build-packages.yml` - Main workflow configuration

### Version Handling

The workflow supports multiple version sources:
1. Git tags (preferred for releases)
2. Environment variable `G4F_VERSION`
3. Manual input in workflow dispatch

Version must follow [PEP 440](https://peps.python.org/pep-0440/) format for PyPI compatibility.

### Troubleshooting

Common issues and solutions:

1. **Build fails**: Check Python version compatibility and dependencies
2. **Version errors**: Ensure version follows PEP 440 format
3. **Missing artifacts**: Check if all build jobs completed successfully
4. **Docker push fails**: Verify Docker Hub credentials are set in repository secrets

### Security Notes

The workflow uses secure practices:
- Trusted action versions
- Environment isolation
- Secret management for credentials
- No hardcoded sensitive data

### Contributing

To improve the build system:
1. Test changes locally first
2. Update documentation
3. Consider backward compatibility
4. Test with multiple Python versions