from ..errors import MissingRequirementsError

try:
    from .server.website import Website
    from .server.backend_api import Backend_Api
    from .server.app import create_app
    import_error = None
except ImportError as e:
    import_error = e

def get_gui_app(demo: bool = False, timeout: int = None):
    if import_error is not None:
        raise MissingRequirementsError(f'Install "gui" requirements | pip install -U g4f[gui]\n{import_error}')
    app = create_app()
    app.demo = demo
    app.timeout = timeout

    site = Website(app)
    for route in site.routes:
        app.add_url_rule(
            route,
            view_func=site.routes[route]['function'],
            methods=site.routes[route]['methods'],
        )

    backend_api  = Backend_Api(app)
    for route in backend_api.routes:
        app.add_url_rule(
            route,
            view_func = backend_api.routes[route]['function'],
            methods   = backend_api.routes[route]['methods'],
        )
    return app

def run_gui(host: str = '0.0.0.0', port: int = 8080, debug: bool = False) -> None:
    config = {
        'host' : host,
        'port' : port,
        'debug': debug
    }

    app = get_gui_app()

    print(f"Running on port {config['port']}")
    app.run(**config)
    print(f"Closing port {config['port']}")
