import requests
import random
import string
import time
import sys
import re
import os


def check_email(mail, logging: bool = False):
    username = mail.split("@")[0]
    domain = mail.split("@")[1]
    reqLink = f"https://www.1secmail.com/api/v1/?action=getMessages&login={username}&domain={domain}"
    req = requests.get(reqLink)
    req.encoding = req.apparent_encoding
    req = req.json()

    length = len(req)

    if logging:
        os.system("cls" if os.name == "nt" else "clear")
        time.sleep(1)
        print("Your temporary mail:", mail)

    if logging and length == 0:
        print(
            "Mailbox is empty. Hold tight. Mailbox is refreshed automatically every 5 seconds.",
        )
    else:
        messages = []
        id_list = []

        for i in req:
            for k, v in i.items():
                if k == "id":
                    id_list.append(v)

        x = "mails" if length > 1 else "mail"

        if logging:
            print(
                f"Mailbox has {length} {x}. (Mailbox is refreshed automatically every 5 seconds.)"
            )

        for i in id_list:
            msgRead = f"https://www.1secmail.com/api/v1/?action=readMessage&login={username}&domain={domain}&id={i}"
            req = requests.get(msgRead)
            req.encoding = req.apparent_encoding
            req = req.json()

            for k, v in req.items():
                if k == "from":
                    sender = v
                if k == "subject":
                    subject = v
                if k == "date":
                    date = v
                if k == "textBody":
                    content = v

            if logging:
                print(
                    "Sender:",
                    sender,
                    "\nTo:",
                    mail,
                    "\nSubject:",
                    subject,
                    "\nDate:",
                    date,
                    "\nContent:",
                    content,
                    "\n",
                )
            messages.append(
                {
                    "sender": sender,
                    "to": mail,
                    "subject": subject,
                    "date": date,
                    "content": content,
                }
            )

        if logging:
            os.system("cls" if os.name == "nt" else "clear")
        return messages


def create_email(custom_domain: bool = False, logging: bool = False):
    domainList = ["1secmail.com", "1secmail.net", "1secmail.org"]
    domain = random.choice(domainList)
    try:
        if custom_domain:
            custom_domain = input(
                "\nIf you enter 'my-test-email' as your domain name, mail address will look like this: 'my-test-email@1secmail.com'"
                "\nEnter the name that you wish to use as your domain name: "
            )

            newMail = f"https://www.1secmail.com/api/v1/?login={custom_domain}&domain={domain}"
            reqMail = requests.get(newMail)
            reqMail.encoding = reqMail.apparent_encoding

            username = re.search(r"login=(.*)&", newMail).group(1)
            domain = re.search(r"domain=(.*)", newMail).group(1)
            mail = f"{username}@{domain}"

            if logging:
                print("\nYour temporary email was created successfully:", mail)
            return mail

        else:
            name = string.ascii_lowercase + string.digits
            random_username = "".join(random.choice(name) for i in range(10))
            newMail = f"https://www.1secmail.com/api/v1/?login={random_username}&domain={domain}"

            reqMail = requests.get(newMail)
            reqMail.encoding = reqMail.apparent_encoding

            username = re.search(r"login=(.*)&", newMail).group(1)
            domain = re.search(r"domain=(.*)", newMail).group(1)
            mail = f"{username}@{domain}"

            if logging:
                print("\nYour temporary email was created successfully:", mail)
            return mail

    except KeyboardInterrupt:
        requests.post(
            "https://www.1secmail.com/mailbox",
            data={
                "action": "deleteMailbox",
                "login": f"{username}",
                "domain": f"{domain}",
            },
        )
        if logging:
            print("\nKeyboard Interrupt Detected! \nTemporary mail was disposed!")
            os.system("cls" if os.name == "nt" else "clear")
