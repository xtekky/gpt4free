from pypasser.exceptions import RecaptchaTokenNotFound, RecaptchaResponseNotFound
from pypasser.session import Session
from pypasser.structs import Proxy
from pypasser.utils import parse_url
from .constants import POST_DATA, BASE_URL, BASE_HEADERS

import re
from typing import Dict, Union

class reCaptchaV3:
    """
    reCaptchaV3 bypass
    -----------------
    Bypass reCaptcha V3 only by sending HTTP requests.
    
    Attributes
    ----------
    anchor_url: str
        The anchor url.
        
    proxy [Optional]: Proxy or Dict,
        Proxy object from `pypasser.structs` or dict (for requests library).

    timeout [Optional]: int or float,
        the number of seconds to wait on a response before timing out.
    """
    def __new__(cls, *args, **kwargs) -> str:
        instance = super(reCaptchaV3, cls).__new__(cls)
        instance.__init__(*args,**kwargs)
        
        cls.session = Session(BASE_URL, BASE_HEADERS, instance.timeout, instance.proxy)
        
        data = parse_url(instance.anchor_url)
        
        # Gets recaptcha token.
        token = cls.get_recaptcha_token(data['endpoint'],
                                        data['params']
                                        )
        
        params = dict(pair.split('=') for pair in data['params'].split('&'))
         
        # Gets recaptcha response.
        post_data = POST_DATA.format(params["v"], token,
                                     params["k"], params["co"])
        
        recaptcha_response = cls.get_recaptcha_response(data['endpoint'],
                                                        f'k={params["k"]}',
                                                        post_data
                                                        )
        
        return recaptcha_response
        
    def __init__(self, anchor_url: str,
                proxy: Union[Proxy, Dict] = None,
                timeout: Union[int, float] = 20):
        
        self.anchor_url = anchor_url
        self.proxy = proxy
        self.timeout = timeout
                   
    def get_recaptcha_token(endpoint: str, params: str) -> str:
        """
        Sends GET request to `anchor URL` to get recaptcha token.
        
        """
        response = reCaptchaV3.session.send_request(
                                f"{endpoint}/anchor", params=params)
        
        results = re.findall(r'"recaptcha-token" value="(.*?)"', response.text)
        if not results:
            raise RecaptchaTokenNotFound()
        
        return results[0]
            

    def get_recaptcha_response(endpoint: str, params: str, data: str) -> str:
        """
        Sends POST request to `reload URL` to get recaptcha response.
        
        """
        response = reCaptchaV3.session.send_request(
                                f"{endpoint}/reload", data=data, params=params)
        
        results = re.findall(r'"rresp","(.*?)"', response.text)
        if not results:
            raise RecaptchaResponseNotFound()
        
        return results[0]