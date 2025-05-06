import asyncio
import sys
import itertools
from g4f.models import glm_4_32b, z1_32b, z1_rumination

async def spinner(msg, event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if event.is_set():
            break
        sys.stdout.write(f'\r{msg} {c}')
        sys.stdout.flush()
        await asyncio.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(msg) + 2) + '\r')  # Clear line

async def test_model(model, token, cookies):
    check_msg = f"Checking {model.name}..."
    done_event = asyncio.Event()
    spinner_task = asyncio.create_task(spinner(check_msg, done_event))
    try:
        messages = [{"role": "user", "content": f"Hello, {model.name}!"}]
        async for _ in model.best_provider.create_async_generator(
            model=model.name, messages=messages, token=token, cookies=cookies
        ):
            break  # Only need to confirm we get a response
        done_event.set()
        await spinner_task
        print(f"\033[92m✔ {model.name} checked\033[0m")  # Green tick
    except Exception as e:
        done_event.set()
        await spinner_task
        print(f"\033[91m✖ {model.name} failed: {e}\033[0m")  # Red cross

async def main():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjI5MjUwM2M5LTcwNjMtNGFmOC1hYzE2LWQ4MzA4M2U1NjA2YSJ9.6bNg5txykNraoZqQI4cD-aVw8yTQmxB68n-WXqj1bOM"
    cookies = {
        "token": token,
        "SERVERID": "9a36118b37582db121a9d682ad6aa4f0|1745177148|1745166091",
        "SERVERCORSID": "9a36118b37582db121a9d682ad6aa4f0|1745177148|1745166091"
    }
    for model in [glm_4_32b, z1_32b, z1_rumination]:
        await test_model(model, token, cookies)

if __name__ == "__main__":
    asyncio.run(main())