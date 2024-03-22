from ..errors import MissingRequirementsError

def run_gui(host: str = '0.0.0.0', port: int = 8080, debug: bool = False) -> None:
    try:
        from .server.app     import app
        from .server.website import Website
        from .server.backend import Backend_Api
    except ImportError:
        raise MissingRequirementsError('Install "flask" package for the gui')

    if debug:
        import g4f
        g4f.debug.logging = True
    config = {
        'host' : host,
        'port' : port,
        'debug': debug
    }
    
    site = Website(app)
    for route in site.routes:
        app.add_url_rule(
            route,
            view_func = site.routes[route]['function'],
            methods   = site.routes[route]['methods'],
        )
    
    backend_api  = Backend_Api(app)
    for route in backend_api.routes:
        app.add_url_rule(
            route,
            view_func = backend_api.routes[route]['function'],
            methods   = backend_api.routes[route]['methods'],
        )
    
    print(f"Running on port {config['port']}")
    app.run(**config)
    print(f"Closing port {config['port']}")
