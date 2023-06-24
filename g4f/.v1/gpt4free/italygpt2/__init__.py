import re
import requests
import hashlib
from fake_useragent import UserAgent
class Account:
    @staticmethod
    def create():
        r=requests.get("https://italygpt.it/",headers=Account._header)
        f=r.text
        tid=re.search('<input type=\"hidden\" name=\"next_id\" id=\"next_id\" value=\"(\w+)\">',f).group(1)
        if len(tid)==0:
            raise RuntimeError("NetWorkError:failed to get id.")
        else:
            Account._tid=tid
        Account._raw="[]"
        return Account
    def next(next_id:str)->str:
        Account._tid=next_id
        return Account._tid
    def get()->str:
        return Account._tid
    _header={
            "Host": "italygpt.it",
            "Referer":"https://italygpt.it/",
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",#UserAgent().random,
            "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language":"zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Upgrade-Insecure-Requests":"1",
            "Sec-Fetch-Dest":"document",
            "Sec-Fetch-Mode":"navigate",
            "Sec-Fetch-Site":"none",
            "Sec-Fetch-User":"?1",
            "Connection":"keep-alive",
            "Alt-Used":"italygpt.it",
            "Pragma":"no-cache",
            "Cache-Control":"no-cache",
            "TE": "trailers"
        }
    def settraw(raws:str):
        Account._raw=raws
        return Account._raw
    def gettraw():
        return Account._raw

class Completion:
    @staticmethod
    def create(
        account_data,
        prompt: str,
        message=False
    ):
        param={
            "prompt":prompt.replace(" ","+"),
            "creative":"off",
            "internet":"false",
            "detailed":"off",
            "current_id":"0",
            "code":"",
            "gpt4":"false",
            "raw_messages":account_data.gettraw(),
            "hash":hashlib.sha256(account_data.get().encode()).hexdigest()
        }
        if(message):
            param["raw_messages"]=str(message)
        r = requests.get("https://italygpt.it/question",headers=account_data._header,params=param,stream=True)
        account_data.next(r.headers["Next_id"])
        account_data.settraw(r.headers["Raw_messages"])
        for chunk in r.iter_content(chunk_size=None):
            r.raise_for_status()
            yield chunk.decode()