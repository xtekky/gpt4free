from .providers.types import ProviderType

logging: bool = False
version_check: bool = True
last_provider: ProviderType = None
last_model: str = None
version: str = None
log_handler: callable = print
logs: list = []

def log(text):
    if logging:
        log_handler(text)