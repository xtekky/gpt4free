import random
import hashlib
import json
import base64
from datetime import datetime, timezone

proof_token_cache: dict = {}

def generate_proof_token(required: bool, seed: str = None, difficulty: str = None, user_agent: str = None, proofTokens: list = None):
    if not required:
        return
    if seed is not None and seed in proof_token_cache:
        return proof_token_cache[seed]

    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    parse_time = now_utc.strftime('%a, %d %b %Y %H:%M:%S GMT')

    if proofTokens:
        config = random.choice(proofTokens)
    else:
        screen = random.choice([3008, 4010, 6000]) * random.choice([1, 2, 4])
        config = [
            screen, parse_time,
            None, 0, user_agent,
            "https://tcr9i.chat.openai.com/v2/35536E1E-65B4-4D96-9D97-6ADB7EFF8147/api.js",
            "dpl=1440a687921de39ff5ee56b92807faaadce73f13","en","en-US",
            None,
            "plugins−[object PluginArray]",
            random.choice(["_reactListeningcfilawjnerp", "_reactListening9ne2dfo1i47", "_reactListening410nzwhan2a"]),
            random.choice(["alert", "ontransitionend", "onprogress"])
        ]

    config[1] = parse_time
    config[4] = user_agent
    config[7] = random.randint(101, 2100)

    diff_len = None if difficulty is None else len(difficulty)
    for i in range(100000):
        config[3] = i
        json_data = json.dumps(config)
        base = base64.b64encode(json_data.encode()).decode()
        hash_value = hashlib.sha3_512((seed or "" + base).encode()).digest()

        if difficulty is None or hash_value.hex()[:diff_len] <= difficulty:
            if seed is None:
                return "gAAAAAC" + base
            proof_token_cache[seed] = "gAAAAAB" + base
            return proof_token_cache[seed]

    fallback_base = base64.b64encode(f'"{seed}"'.encode()).decode()
    return "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base
