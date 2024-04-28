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
    # Convert UTC time to Eastern Time
    now_et = now_utc.astimezone(timezone(timedelta(hours=-5)))

    parse_time = now_et.strftime('%a, %d %b %Y %H:%M:%S GMT')

    config = [core + screen, parse_time, 4294705152, 0, user_agent]

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
