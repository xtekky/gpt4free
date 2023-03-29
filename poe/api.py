# ading2210/poe-api: a reverse engineered Python API wrapepr for Quora's Poe
# Copyright (C) 2023 ading2210

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import requests
import re
import json
import random
import logging
import time
import queue
import threading
import websocket
from pathlib import Path
from urllib.parse import urlparse

parent_path = Path(__file__).resolve().parent
queries_path = parent_path / "graphql"
queries = {}

logging.basicConfig()
logger = logging.getLogger()

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"


def load_queries():
    for path in queries_path.iterdir():
        if path.suffix != ".graphql":
            continue
        with open(path) as f:
            queries[path.stem] = f.read()


def generate_payload(query_name, variables):
    return {
        "query": queries[query_name],
        "variables": variables
    }


def request_with_retries(method, *args, **kwargs):
    attempts = kwargs.get("attempts") or 10
    url = args[0]
    for i in range(attempts):
        r = method(*args, **kwargs)
        if r.status_code == 200:
            return r
        logger.warn(
            f"Server returned a status code of {r.status_code} while downloading {url}. Retrying ({i+1}/{attempts})...")

    raise RuntimeError(f"Failed to download {url} too many times.")


class Client:
    gql_url = "https://poe.com/api/gql_POST"
    gql_recv_url = "https://poe.com/api/receive_POST"
    home_url = "https://poe.com"
    settings_url = "https://poe.com/api/settings"

    formkey = ""
    next_data = {}
    bots = {}
    active_messages = {}
    message_queues = {}
    ws = None
    ws_connected = False

    def __init__(self, token, proxy=None):
        self.proxy = proxy
        self.session = requests.Session()

        if proxy:
            self.session.proxies = {
                "http": self.proxy,
                "https": self.proxy
            }
            logger.info(f"Proxy enabled: {self.proxy}")

        self.session.cookies.set("p-b", token, domain="poe.com")
        self.headers = {
            "User-Agent": user_agent,
            "Referrer": "https://poe.com/",
            "Origin": "https://poe.com",
        }
        self.ws_domain = f"tch{random.randint(1, 1e6)}"

        self.session.headers.update(self.headers)
        self.next_data = self.get_next_data()
        self.channel = self.get_channel_data()
        self.connect_ws()
        self.bots = self.get_bots()
        self.bot_names = self.get_bot_names()

        self.gql_headers = {
            "poe-formkey": self.formkey,
            "poe-tchannel": self.channel["channel"],
        }
        self.gql_headers = {**self.gql_headers, **self.headers}
        self.subscribe()

    def get_next_data(self):
        logger.info("Downloading next_data...")

        r = request_with_retries(self.session.get, self.home_url)
        json_regex = r'<script id="__NEXT_DATA__" type="application\/json">(.+?)</script>'
        json_text = re.search(json_regex, r.text).group(1)
        next_data = json.loads(json_text)

        self.formkey = next_data["props"]["formkey"]
        self.viewer = next_data["props"]["pageProps"]["payload"]["viewer"]

        return next_data

    def get_bots(self):
        viewer = self.next_data["props"]["pageProps"]["payload"]["viewer"]
        if not "availableBots" in viewer:
            raise RuntimeError("Invalid token.")
        bot_list = viewer["availableBots"]

        bots = {}
        for bot in bot_list:
            url = f'https://poe.com/_next/data/{self.next_data["buildId"]}/{bot["displayName"].lower()}.json'
            logger.info("Downloading "+url)

            r = request_with_retries(self.session.get, url)

            chat_data = r.json()[
                "pageProps"]["payload"]["chatOfBotDisplayName"]
            bots[chat_data["defaultBotObject"]["nickname"]] = chat_data

        return bots

    def get_bot_names(self):
        bot_names = {}
        for bot_nickname in self.bots:
            bot_obj = self.bots[bot_nickname]["defaultBotObject"]
            bot_names[bot_nickname] = bot_obj["displayName"]
        return bot_names

    def get_channel_data(self, channel=None):
        logger.info("Downloading channel data...")
        r = request_with_retries(self.session.get, self.settings_url)
        data = r.json()

        self.formkey = data["formkey"]
        return data["tchannelData"]

    def get_websocket_url(self, channel=None):
        if channel is None:
            channel = self.channel
        query = f'?min_seq={channel["minSeq"]}&channel={channel["channel"]}&hash={channel["channelHash"]}'
        return f'wss://{self.ws_domain}.tch.{channel["baseHost"]}/up/{channel["boxName"]}/updates'+query

    def send_query(self, query_name, variables):
        # print(f'send_query: {query_name} {variables}')
        
        for i in range(20):
            payload = generate_payload(query_name, variables)
            # print(f'query_payload: {query_name} {variables}')
            r = request_with_retries(
                self.session.post, self.gql_url, json=payload, headers=self.gql_headers)
            data = r.json()
            if data["data"] == None:
                logger.warn(
                    f'{query_name} returned an error: {data["errors"][0]["message"]} | Retrying ({i+1}/20)')
                time.sleep(2)
                continue

            return r.json()

        raise RuntimeError(f'{query_name} failed too many times.')

    def subscribe(self):
        logger.info("Subscribing to mutations")
        result = self.send_query("SubscriptionsMutation", {
            "subscriptions": [
                {
                    "subscriptionName": "messageAdded",
                    "query": queries["MessageAddedSubscription"]
                },
                {
                    "subscriptionName": "viewerStateUpdated",
                    "query": queries["ViewerStateUpdatedSubscription"]
                }
            ]
        })

    def ws_run_thread(self):
        kwargs = {}
        if self.proxy:
            proxy_parsed = urlparse(self.proxy)
            kwargs = {
                "proxy_type": proxy_parsed.scheme,
                "http_proxy_host": proxy_parsed.hostname,
                "http_proxy_port": proxy_parsed.port
            }

        self.ws.run_forever(**kwargs)

    def connect_ws(self):
        self.ws = websocket.WebSocketApp(
            self.get_websocket_url(),
            header={"User-Agent": user_agent},
            on_message=self.on_message,
            on_open=self.on_ws_connect,
            on_error=self.on_ws_error
        )
        t = threading.Thread(target=self.ws_run_thread, daemon=True)
        t.start()
        while not self.ws_connected:
            time.sleep(0.01)

    def disconnect_ws(self):
        if self.ws:
            self.ws.close()
        self.ws_connected = False

    def on_ws_connect(self, ws):
        self.ws_connected = True

    def on_ws_error(self, ws, error):
        logger.warn(f"Websocket returned error: {error}")
        self.disconnect_ws()
        self.connect_ws()

    def on_message(self, ws, msg):
        data = json.loads(msg)
        message = json.loads(data["messages"][0])[
            "payload"]["data"]["messageAdded"]

        copied_dict = self.active_messages.copy()
        for key, value in copied_dict.items():
            # add the message to the appropriate queue
            if value == message["messageId"] and key in self.message_queues:
                self.message_queues[key].put(message)
                return

            # indicate that the response id is tied to the human message id
            elif key != "pending" and value == None and message["state"] != "complete":
                self.active_messages[key] = message["messageId"]
                self.message_queues[key].put(message)

    def send_message(self, chatbot, message, with_chat_break=False, timeout=20):
        # if there is another active message, wait until it has finished sending
        while None in self.active_messages.values():
            time.sleep(0.01)

        # None indicates that a message is still in progress
        self.active_messages["pending"] = None

        logger.info(f"Sending message to {chatbot}: {message}")

        message_data = self.send_query("AddHumanMessageMutation", {
            "bot": chatbot,
            "query": message,
            "chatId": self.bots[chatbot]["chatId"],
            "source": None,
            "withChatBreak": with_chat_break
        })
        del self.active_messages["pending"]

        if not message_data["data"]["messageCreateWithStatus"]["messageLimit"]["canSend"]:
            raise RuntimeError(f"Daily limit reached for {chatbot}.")
        try:
            human_message = message_data["data"]["messageCreateWithStatus"]
            human_message_id = human_message["message"]["messageId"]
        except TypeError:
            raise RuntimeError(
                f"An unknown error occured. Raw response data: {message_data}")

        # indicate that the current message is waiting for a response
        self.active_messages[human_message_id] = None
        self.message_queues[human_message_id] = queue.Queue()

        last_text = ""
        message_id = None
        while True:
            try:
                message = self.message_queues[human_message_id].get(
                    timeout=timeout)
            except queue.Empty:
                del self.active_messages[human_message_id]
                del self.message_queues[human_message_id]
                raise RuntimeError("Response timed out.")

            # only break when the message is marked as complete
            if message["state"] == "complete":
                if last_text and message["messageId"] == message_id:
                    break
                else:
                    continue

            # update info about response
            message["text_new"] = message["text"][len(last_text):]
            last_text = message["text"]
            message_id = message["messageId"]

            yield message

        del self.active_messages[human_message_id]
        del self.message_queues[human_message_id]

load_queries()
