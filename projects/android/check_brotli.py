"""Verify that a Chaquopy APK bundles Brotli for every supported ABI."""

import io
import sys
import zipfile


def check_apk(apk_path):
    with zipfile.ZipFile(apk_path) as apk:
        common = "assets/chaquopy/requirements-common.imy"
        with zipfile.ZipFile(io.BytesIO(apk.read(common))) as requirements:
            if "brotli.pyc" not in requirements.namelist():
                raise RuntimeError(f"Missing brotli.pyc in {common}")

        for abi in ("arm64-v8a", "x86_64"):
            path = f"assets/chaquopy/requirements-{abi}.imy"
            with zipfile.ZipFile(io.BytesIO(apk.read(path))) as requirements:
                if "_brotli.so" not in requirements.namelist():
                    raise RuntimeError(f"Missing _brotli.so in {path}")


if __name__ == "__main__":
    check_apk(sys.argv[1])
    print("Brotli is bundled for arm64-v8a and x86_64")
