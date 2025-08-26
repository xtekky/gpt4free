#!/bin/bash
# Validation test for Nuitka build system

set -e

echo "=== G4F Nuitka Build Validation Test ==="
echo "Testing the new Nuitka-based build system"

# Test 1: Check if g4f_cli.py loads correctly
echo "Test 1: Verifying g4f_cli.py entry point..."
if python g4f_cli.py --help > /dev/null 2>&1; then
    echo "✓ g4f_cli.py works correctly"
else
    echo "✗ g4f_cli.py failed"
    exit 1
fi

# Test 2: Check if Nuitka is available
echo "Test 2: Verifying Nuitka installation..."
if python -m nuitka --version > /dev/null 2>&1; then
    echo "✓ Nuitka is installed and working"
else
    echo "✗ Nuitka is not available"
    exit 1
fi

# Test 3: Check if build script exists and is executable
echo "Test 3: Verifying build script..."
if [[ -x "scripts/build-nuitka.sh" ]]; then
    echo "✓ Build script is executable"
else
    echo "✗ Build script is missing or not executable"
    exit 1
fi

# Test 4: Check if workflow includes Nuitka
echo "Test 4: Verifying GitHub Actions workflow..."
if grep -q "nuitka" .github/workflows/build-packages.yml; then
    echo "✓ Workflow uses Nuitka"
else
    echo "✗ Workflow doesn't use Nuitka"
    exit 1
fi

# Test 5: Verify architecture support in workflow
echo "Test 5: Verifying architecture matrix in workflow..."
if grep -q "matrix:" .github/workflows/build-packages.yml && grep -q "architecture:" .github/workflows/build-packages.yml; then
    echo "✓ Architecture matrix is present"
else
    echo "✗ Architecture matrix is missing"
    exit 1
fi

echo "=== All Tests Passed! ==="
echo "The Nuitka build system is properly configured."
echo ""
echo "Next steps:"
echo "1. Test the build in CI environment"
echo "2. Verify executable quality and performance"
echo "3. Consider adding ARM64 Linux builds with dedicated runners"