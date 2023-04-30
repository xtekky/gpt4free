import forefront


def create_account(logging_enabled=True):
    """Create a Forefront account and return the token."""
    token = forefront.Account.create(logging=logging_enabled)
    return token


def get_response(token, prompt, model_name):
    """Generate and print responses using the Forefront API."""
    streaming_completion = forefront.StreamingCompletion.create(
        token=token, prompt=prompt, model=model_name
    )

    for response in streaming_completion:
        completion_text = response.completion.choices[0].text
        print(completion_text, end='')


def main():
    # Create an account
    token = create_account(logging=True)
    print(f"Account created with token: {token}")

    # Generate and print responses using GPT-4 model
    prompt = "hello world"
    model_name = "gpt-4"
    get_response(token, prompt, model_name)


if __name__ == "__main__":
    main()
