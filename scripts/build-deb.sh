#!/bin/bash
# Debian package build script for g4f

set -e

PACKAGE_NAME="g4f"
VERSION="${G4F_VERSION:-0.0.0-dev}"
ARCHITECTURE="${ARCH:-amd64}"
MAINTAINER="Tekky <support@g4f.ai>"
DESCRIPTION="The official gpt4free repository"
LONG_DESCRIPTION="Various collection of powerful language models"

# Clean up any previous builds
rm -rf debian/

# Create package directory structure
mkdir -p debian/${PACKAGE_NAME}/DEBIAN
mkdir -p debian/${PACKAGE_NAME}/usr/bin
mkdir -p debian/${PACKAGE_NAME}/usr/lib/python3/dist-packages
mkdir -p debian/${PACKAGE_NAME}/usr/share/doc/${PACKAGE_NAME}
mkdir -p debian/${PACKAGE_NAME}/usr/share/applications

# Create control file
cat > debian/${PACKAGE_NAME}/DEBIAN/control << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: python
Priority: optional
Architecture: ${ARCHITECTURE}
Essential: no
Maintainer: ${MAINTAINER}
Description: ${DESCRIPTION}
 ${LONG_DESCRIPTION}
Depends: python3 (>= 3.10), python3-pip, python3-aiohttp, python3-requests
Homepage: https://github.com/xtekky/gpt4free
EOF

# Create postinst script
cat > debian/${PACKAGE_NAME}/DEBIAN/postinst << 'EOF'
#!/bin/bash
set -e

# Install Python dependencies
pip3 install --break-system-packages aiohttp requests brotli pycryptodome nest_asyncio

# Make g4f command available
if [ ! -L /usr/local/bin/g4f ]; then
    ln -s /usr/bin/g4f /usr/local/bin/g4f
fi

echo "g4f installed successfully"
echo "Usage: g4f --help"
EOF

# Create prerm script
cat > debian/${PACKAGE_NAME}/DEBIAN/prerm << 'EOF'
#!/bin/bash
set -e

# Remove symlink if it exists
if [ -L /usr/local/bin/g4f ]; then
    rm -f /usr/local/bin/g4f
fi
EOF

# Make scripts executable
chmod 755 debian/${PACKAGE_NAME}/DEBIAN/postinst
chmod 755 debian/${PACKAGE_NAME}/DEBIAN/prerm

# Install the package files
export PYTHONPATH=""
python3 setup.py install --root=debian/${PACKAGE_NAME} --prefix=/usr --install-lib=/usr/lib/python3/dist-packages --install-scripts=/usr/bin

# Create documentation
cp README.md debian/${PACKAGE_NAME}/usr/share/doc/${PACKAGE_NAME}/
cp LICENSE debian/${PACKAGE_NAME}/usr/share/doc/${PACKAGE_NAME}/copyright
gzip -9 debian/${PACKAGE_NAME}/usr/share/doc/${PACKAGE_NAME}/README.md

# Create desktop file
cat > debian/${PACKAGE_NAME}/usr/share/applications/${PACKAGE_NAME}.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=g4f
Comment=${DESCRIPTION}
Exec=/usr/bin/g4f
Icon=application-x-executable
Terminal=false
Categories=Development;Network;
EOF

# Fix permissions
find debian/${PACKAGE_NAME} -type d -exec chmod 755 {} \;
find debian/${PACKAGE_NAME} -type f -exec chmod 644 {} \;
chmod 755 debian/${PACKAGE_NAME}/usr/bin/g4f
chmod 755 debian/${PACKAGE_NAME}/DEBIAN/postinst
chmod 755 debian/${PACKAGE_NAME}/DEBIAN/prerm

# Calculate installed size
INSTALLED_SIZE=$(du -sk debian/${PACKAGE_NAME}/usr | cut -f1)

# Add installed size to control file
echo "Installed-Size: ${INSTALLED_SIZE}" >> debian/${PACKAGE_NAME}/DEBIAN/control

# Build the package
dpkg-deb --build debian/${PACKAGE_NAME} ${PACKAGE_NAME}-${VERSION}-${ARCHITECTURE}.deb

echo "Debian package created: ${PACKAGE_NAME}-${VERSION}-${ARCHITECTURE}.deb"