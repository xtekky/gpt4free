from g4f.api import run_api

if __name__ == "__main__":
    run_api(
        host='0.0.0.0',  # Listen on all network interfaces
        port=1337,       # Default port
        debug=True       # Enable debug mode
    )
