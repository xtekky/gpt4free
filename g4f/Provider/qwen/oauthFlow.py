import asyncio
import webbrowser
import time
from .qwenOAuth2 import generatePKCEPair, QwenOAuth2Client

# Configuration
AUTHORIZATION_URL = "https://chat.qwen.ai/api/v1/oauth2/device/code"
TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
SCOPES = "openid profile email model.completion"

# Local redirect URL for redirect-based flow (if used)
REDIRECT_URI = "http://localhost:8080/callback"

async def launch_browser_for_oauth():
    # Generate PKCE parameters
    pkce_pair = generatePKCEPair()
    code_verifier = pkce_pair['code_verifier']
    code_challenge = pkce_pair['code_challenge']

    # Initialize OAuth client
    client = QwenOAuth2Client()

    # Request device code
    device_auth = await client.requestDeviceAuthorization({
        "scope": SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })

    # Check device auth success
    if not isinstance(device_auth, dict) or "device_code" not in device_auth:
        print("Failed to receive device code")
        return

    # Show user instructions
    print("Please visit the following URL to authorize:")
    print(device_auth.get("verification_uri_complete") or device_auth["verification_uri"])
    # Attempt to automatically open the URL
    url_to_open = device_auth.get("verification_uri_complete") or device_auth["verification_uri"]
    try:
        webbrowser.open(url_to_open)
    except:
        print(f"Open the URL manually in your browser: {url_to_open}")

    # Start polling for token
    device_code = device_auth["device_code"]
    expires_in = device_auth.get("expires_in", 1800)  # default 30 min
    start_time = time.time()

    print("Waiting for authorization... Press Ctrl+C to cancel.")

    while True:
        if time.time() - start_time > expires_in:
            print("Authorization timed out.")
            break

        # Poll for token
        token_response = await client.pollDeviceToken({
            "device_code": device_code,
            "code_verifier": code_verifier,
        })

        if isinstance(token_response, dict):
            if "status" in token_response and token_response["status"] == "pending":
                print(".", end="", flush=True)
                await asyncio.sleep(2)  # polling interval
                continue
            elif "access_token" in token_response:
                # Success
                print("\nAuthorization successful.")
                print("Access Token:", token_response["access_token"])
                # Save token_response to a file or config
                credentials = {
                    "access_token": token_response["access_token"],
                    "token_type": token_response["token_type"],
                    "refresh_token": token_response.get("refresh_token"),
                    "resource_url": token_response.get("resource_url"),
                    "expiry_date": int(time.time() * 1000) + token_response.get("expires_in", 0) * 1000,
                }
                await client.sharedManager.saveCredentialsToFile(credentials)
                print(f"Credentials saved to: {client.sharedManager.getCredentialFilePath()}")
                return
            else:
                print(f"\nError during polling: {token_response}")
                break
        else:
            print(f"\nUnexpected response: {token_response}")
            break

# Run the entire process
async def main():
    await launch_browser_for_oauth()

if __name__ == "__main__":
    asyncio.run(main())
