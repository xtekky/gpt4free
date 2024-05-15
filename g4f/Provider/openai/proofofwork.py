import random
import hashlib
import json
import base64
from datetime import datetime, timedelta, timezone

def generate_proof_token(required: bool, seed: str, difficulty: str, user_agent: str):
    if not required:
        return

    cores = [8, 12, 16, 24]
    screens = [3000, 4000, 6000]

    core = random.choice(cores)
    screen = random.choice(screens)

    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    parse_time = now_utc.strftime('%a, %d %b %Y %H:%M:%S GMT')

    config = [core + screen, parse_time, None, 0, user_agent, "https://tcr9i.chat.openai.com/v2/35536E1E-65B4-4D96-9D97-6ADB7EFF8147/api.js","dpl=53d243de46ff04dadd88d293f088c2dd728f126f","en","en-US",442,"plugins−[object PluginArray]","","alert"]

    diff_len = len(difficulty) // 2

    for i in range(100000):
        config[3] = i
        json_data = json.dumps(config)
        base = base64.b64encode(json_data.encode()).decode()
        hash_value = hashlib.sha3_512((seed + base).encode()).digest()

        if hash_value.hex()[:diff_len] <= difficulty:
            result = "gAAAAAB" + base
            return result

    fallback_base = base64.b64encode(f'"{seed}"'.encode()).decode()
    return "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base
