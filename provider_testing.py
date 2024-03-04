import g4f
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
from requests.exceptions import SSLError, HTTPError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Prompt to be used for testing
prompt = "say HI"

response_times = {}

def run_provider(provider: g4f.Provider.BaseProvider):
    start_time = time.time()
    try:
        response = g4f.ChatCompletion.create(
            model=g4f.models.default,
            messages=[{"role": "user", "content": prompt }],
            provider=provider,
            auth=True,
        )
        end_time = time.time()
        response_time = end_time - start_time
        if response:
            logging.info(f"✅ {provider.__name__}: {response}")
            logging.info(f"Response time (seconds): {response_time}")
            response_times[provider.__name__] = response_time
        else:
            logging.warning(f"🚧 No response from {provider.__name__}")
    except SSLError:
        logging.error(f"❌ {provider.__name__}: SSL Certificate Verification Error")
    except HTTPError as e:
        logging.error(f"❌ {provider.__name__}: HTTP Error {e.response.status_code}")
    except Exception as e:
        logging.error(f"❌ {provider.__name__}: {e}")

def run_all():
    # Filtering providers that are working
    working_providers = [
        provider for provider in g4f.Provider.__providers__
        if hasattr(provider, 'working') and provider.working
    ]

    # Adjusting the number of workers based on the number of providers and system resources
    num_workers = min(30, len(working_providers))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run_provider, working_providers)

    # Sorting providers by response time
    sorted_response_times = sorted(response_times.items(), key=lambda x: x[1])
    logging.info("\n ----------------- \n")
    logging.info("Ranking of the providers by the response time:")
    for i, (provider, time) in enumerate(sorted_response_times, start=1):
        logging.info(f"{i}. {provider}: {time} seconds")

def main():
    try:
        run_all()
    except SSLError:
        logging.exception("An SSL Certificate Verification Error occurred.")
        sys.exit(1)
    except HTTPError as e:
        logging.exception(f"An HTTP Error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
