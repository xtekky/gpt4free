import browser_cookie3


class Utils:
    browsers = [ 
        browser_cookie3.chrome,   # 62.74% market share
        browser_cookie3.safari,   # 24.12% market share
        browser_cookie3.firefox,  #  4.56% market share
        browser_cookie3.edge,     #  2.85% market share 
        browser_cookie3.opera,    #  1.69% market share
        browser_cookie3.brave,    #  0.96% market share
        browser_cookie3.opera_gx, #  0.64% market share
        browser_cookie3.vivaldi,  #  0.32% market share
    ]

    def get_cookies(domain: str, setName: str = None, setBrowser: str = False) -> dict:
        cookies = {}
        
        if setBrowser != False:
            for browser in Utils.browsers:
                if browser.__name__ == setBrowser:
                    try:
                        for c in browser(domain_name=domain):
                            if c.name not in cookies:
                                cookies = cookies | {c.name: c.value} 
                    
                    except Exception as e:
                        pass
        
        else:
            for browser in Utils.browsers:
                try:
                    for c in browser(domain_name=domain):
                        if c.name not in cookies:
                            cookies = cookies | {c.name: c.value} 
                
                except Exception as e:
                    pass
        
        if setName:
            try:
                return {setName: cookies[setName]}
            
            except ValueError:
                print(f'Error: could not find {setName} cookie in any browser.')
                exit(1)
        
        else:
            return cookies
