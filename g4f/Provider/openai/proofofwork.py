import random
import hashlib
import json
import base64
from datetime import datetime, timezone

def generate_proof_token(required: bool, seed: str = "", difficulty: str = "", user_agent: str = None, proof_token: str = None):
    if not required:
        return

    if proof_token is None:
        screen = random.choice([3008, 4010, 6000]) * random.choice([1, 2, 4])
        # Get current UTC time
        now_utc = datetime.now(timezone.utc)
        parse_time = now_utc.strftime('%a, %d %b %Y %H:%M:%S GMT')
        proof_token = [
            screen, parse_time,
            None, 0, user_agent,
            "https://tcr9i.chat.openai.com/v2/35536E1E-65B4-4D96-9D97-6ADB7EFF8147/api.js",
            "dpl=1440a687921de39ff5ee56b92807faaadce73f13","en","en-US",
            None,
            "plugins−[object PluginArray]",
            random.choice(["_reactListeningcfilawjnerp", "_reactListening9ne2dfo1i47", "_reactListening410nzwhan2a"]),
            random.choice(["alert", "ontransitionend", "onprogress"])
        ]

    diff_len = len(difficulty)
    for i in range(100000):
        proof_token[3] = i
        json_data = json.dumps(proof_token)
        base = base64.b64encode(json_data.encode()).decode()
        hash_value = hashlib.sha3_512((seed + base).encode()).digest()

        if hash_value.hex()[:diff_len] <= difficulty:
            return "gAAAAAB" + base

    fallback_base = base64.b64encode(f'"{seed}"'.encode()).decode()
    return "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base
