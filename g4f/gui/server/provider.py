import g4f

def get_provider(provider: str) -> g4f.Provider.BaseProvider:
    
    if isinstance(provider, str):
        print(provider)
        if provider == 'g4f.Provider.Auto':
            return None
        
        if provider in g4f.Provider.ProviderUtils.convert:
            return g4f.Provider.ProviderUtils.convert[provider]
        
        else:
            return None
        
    else:
        return None
