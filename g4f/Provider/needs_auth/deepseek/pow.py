from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional


try:
    import numpy
    import wasmtime

    has_wasmtime_and_numpy = True
except ImportError:
    has_wasmtime_and_numpy = False


WASM_PATH = str(Path(__file__).with_name("pow_solver.wasm"))
DEEPSEEK_POW_ALGORITHM = "DeepSeekHashV1"


class DeepSeekHash:
    """Custom SHA3 hash solver using WebAssembly."""

    def __init__(self):
        self.instance = None
        self.memory = None
        self.store = None

    def init(self, wasm_path: str):
        if not has_wasmtime_and_numpy:
            raise ImportError("wasmtime and numpy are required for PoW solving")

        if not Path(wasm_path).exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_path}")

        engine = wasmtime.Engine()
        with open(wasm_path, "rb") as file:
            wasm_bytes = file.read()
        module = wasmtime.Module(engine, wasm_bytes)

        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()

        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]
        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](
            self.store, length, 1
        )

        memory_view = self.memory.data_ptr(self.store)
        for index, byte in enumerate(encoded):
            memory_view[ptr + index] = byte
        return ptr, length

    def calculate_hash(
            self,
            algorithm: str,
            challenge: str,
            salt: str,
            difficulty: int,
            expire_at: int,
    ) -> Optional[int]:
        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](
            self.store, -16
        )

        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)

            self.instance.exports(self.store)["wasm_solve"](
                self.store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty),
            )

            memory_view = self.memory.data_ptr(self.store)
            status = int.from_bytes(
                bytes(memory_view[retptr: retptr + 4]),
                byteorder="little",
                signed=True,
            )
            if status == 0:
                return None

            value_bytes = bytes(memory_view[retptr + 8: retptr + 16])
            value = numpy.frombuffer(value_bytes, dtype=numpy.float64)[0]
            return int(value)
        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](
                self.store, 16
            )


class DeepSeekPOW:
    """Proof-of-work solver for DeepSeek challenges."""

    def __init__(self):
        self.hasher = DeepSeekHash().init(WASM_PATH)

    def solve_challenge(self, config: dict) -> str:
        answer = self.hasher.calculate_hash(
            config["algorithm"],
            config["challenge"],
            config["salt"],
            config["difficulty"],
            config["expire_at"],
        )
        if answer is None:
            raise RuntimeError("DeepSeek PoW solver returned no answer")

        result = {
            "algorithm": config["algorithm"],
            "challenge": config["challenge"],
            "salt": config["salt"],
            "answer": answer,
            "signature": config["signature"],
            "target_path": config.get("target_path", ""),
        }
        return base64.b64encode(json.dumps(result).encode()).decode()
