import os

from .server.app import app
from .server.api import Api
from .server.website import Website
from .server.backend import Backend_Api
from typing import List, Union
from .. import BaseProvider


def run_gui(host: str = '0.0.0.0',
            port: int = 80,
            debug: bool = False,
            apiStatus: bool = False,
            list_ignored_providers: List[Union[str, BaseProvider]] = None
            ) -> None:
    config = {
        'host': host,
        'port': port,
        'debug': debug
    }
    env = {}
    proxy = os.environ.get('PROXY', None)
    socks5 = os.environ.get('SOCKS5', None)
    timeout = os.environ.get('TIMEOUT', 120)
    if proxy:
        print(f'Proxy: {proxy}')
        env['proxy'] = proxy
    if socks5:
        print(f'Socks5: {socks5}')
        env['socks5'] = socks5
    if timeout:
        print(f'Timeout: {timeout}')
        env['timeout'] = timeout

    site = Website(app)
    for route in site.routes:
        app.add_url_rule(
            route,
            view_func=site.routes[route]['function'],
            methods=site.routes[route]['methods'],
        )

    backend_api = Backend_Api(app)
    for route in backend_api.routes:
        app.add_url_rule(
            route,
            view_func=backend_api.routes[route]['function'],
            methods=backend_api.routes[route]['methods'],
        )

    if apiStatus:
        api = Api(app, env, list_ignored_providers)
        for route in api.routes:
            app.add_url_rule(
                route,
                view_func=api.routes[route]['function'],
                methods=api.routes[route]['methods'],
            )

    print(f"Running on port {config['port']}")
    app.run(**config)
    print(f"Closing port {config['port']}")
