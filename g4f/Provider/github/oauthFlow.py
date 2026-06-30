import asyncio
import webbrowser
import time

from .githubOAuth2 import GithubOAuth2Client, GITHUB_COPILOT_SCOPE


async def launch_browser_for_oauth(client_id: str = None):
    """
    Perform GitHub OAuth device flow for authentication.
    
    This function:
    1. Requests a device code from GitHub
    2. Opens a browser for the user to authenticate
    3. Polls for the access token
    4. Saves the credentials
    
    Args:
        client_id: Optional custom client ID (defaults to Copilot VS Code extension)
    """
    # Initialize OAuth client
    client = GithubOAuth2Client(client_id) if client_id else GithubOAuth2Client()

    # Request device code
    print("Requesting device authorization from GitHub...")
    device_auth = await client.requestDeviceAuthorization({
        "scope": GITHUB_COPILOT_SCOPE,
    })

    # Check device auth success
    if not isinstance(device_auth, dict) or "device_code" not in device_auth:
        print("Failed to receive device code")
        return None

    # Show user instructions
    user_code = device_auth.get("user_code")
    verification_uri = device_auth.get("verification_uri", "https://github.com/login/device")
    
    print("\n" + "=" * 60)
    print("GitHub Copilot Authorization")
    print("=" * 60)
    print(f"\nPlease visit: {verification_uri}")
    print(f"Enter code: {user_code}")
    print("=" * 60 + "\n")
    
    # Attempt to automatically open the URL
    try:
        webbrowser.open(verification_uri)
        print("Browser opened automatically.")
    except Exception:
        print(f"Please open the URL manually in your browser: {verification_uri}")

    # Start polling for token
    device_code = device_auth["device_code"]
    expires_in = device_auth.get("expires_in", 900)  # default 15 min
    interval = device_auth.get("interval", 5)  # default 5 seconds
    start_time = time.time()

    print("\nWaiting for authorization... Press Ctrl+C to cancel.")

    while True:
        if time.time() - start_time > expires_in:
            print("\nAuthorization timed out. Please try again.")
            return None

        # Poll for token
        token_response = await client.pollDeviceToken({
            "device_code": device_code,
        })

        if isinstance(token_response, dict):
            if token_response.get("status") == "pending":
                if token_response.get("slowDown"):
                    interval += 5  # Increase interval as requested by GitHub
                print(".", end="", flush=True)
                await asyncio.sleep(interval)
                continue
            elif "access_token" in token_response:
                # Success
                print("\n\n✓ Authorization successful!")
                
                # Save credentials
                credentials = {
                    "access_token": token_response["access_token"],
                    "token_type": token_response.get("token_type", "bearer"),
                    "scope": token_response.get("scope", ""),
                    # GitHub tokens don't expire, but we can set a far future date
                    "expiry_date": int(time.time() * 1000) + (365 * 24 * 60 * 60 * 1000),  # 1 year
                }
                
                await client.sharedManager.saveCredentialsToFile(credentials)
                print(f"Credentials saved to: {client.sharedManager.getCredentialFilePath()}")
                
                return credentials
            else:
                print(f"\nError during polling: {token_response}")
                return None
        else:
            print(f"\nUnexpected response: {token_response}")
            return None


async def main():
    """Run the OAuth flow."""
    await launch_browser_for_oauth()


if __name__ == "__main__":
    asyncio.run(main())
