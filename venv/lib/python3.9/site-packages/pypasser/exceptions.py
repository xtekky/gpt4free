class RecaptchaTokenNotFound(Exception):
    def __init__(self):
        super().__init__('Recaptcha token not found.')

class RecaptchaResponseNotFound(Exception):
    def __init__(self):
        super().__init__('Recaptcha response not found.')
        
class ConnectionError(Exception):
    pass

class IpBlock(Exception):
    def __init__(self):
        super().__init__('Too many tries for solving reCaptchaV2 using speech to text, take a break or change your ip.')