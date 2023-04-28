__all__ = ["AsyncCurl"]


import asyncio
import os
from typing import Any

from ._wrapper import ffi, lib  # type: ignore
from .const import CurlMOpt
from .curl import Curl, CurlError, CurlInfo

DEFAULT_CACERT = os.path.join(os.path.dirname(__file__), "cacert.pem")

CURL_POLL_NONE = 0
CURL_POLL_IN = 1
CURL_POLL_OUT = 2
CURL_POLL_INOUT = 3
CURL_POLL_REMOVE = 4

CURL_SOCKET_TIMEOUT = -1
CURL_SOCKET_BAD = -1

CURL_CSELECT_IN = 0x01
CURL_CSELECT_OUT = 0x02
CURL_CSELECT_ERR = 0x04

CURLMSG_DONE = 1


@ffi.def_extern()
def timer_function(curlm, timeout_ms: int, clientp: Any):
    """
    see: https://curl.se/libcurl/c/CURLMOPT_TIMERFUNCTION.html
    """
    async_curl = ffi.from_handle(clientp)
    # print("time out in %sms" % timeout_ms)
    if timeout_ms == -1:
        for timer in async_curl._timers:
            timer.cancel()
        async_curl._timers = []
    else:
        timer = async_curl.loop.call_later(
            timeout_ms / 1000,
            async_curl.process_data,
            CURL_SOCKET_TIMEOUT,  # -1
            CURL_POLL_NONE,  # 0
        )
        async_curl._timers.append(timer)


@ffi.def_extern()
def socket_function(curl, sockfd: int, what: int, clientp: Any, data: Any):
    async_curl = ffi.from_handle(clientp)
    loop = async_curl.loop

    # Always remove and readd fd
    if sockfd in async_curl._sockfds:
        loop.remove_reader(sockfd)
        loop.remove_writer(sockfd)

    if what & CURL_POLL_IN:
        loop.add_reader(sockfd, async_curl.process_data, sockfd, CURL_CSELECT_IN)
        async_curl._sockfds.add(sockfd)
    if what & CURL_POLL_OUT:
        loop.add_writer(sockfd, async_curl.process_data, sockfd, CURL_CSELECT_OUT)
        async_curl._sockfds.add(sockfd)
    if what & CURL_POLL_REMOVE:
        async_curl._sockfds.remove(sockfd)


class AsyncCurl:
    def __init__(self, cacert: str = DEFAULT_CACERT, loop=None):
        self._curlm = lib.curl_multi_init()
        self._cacert = cacert
        self._curl2future = {}  # curl to future map
        self._curl2curl = {}  # c curl to Curl
        self._sockfds = set()  # sockfds
        self.loop = loop if loop is not None else asyncio.get_running_loop()
        self._checker = self.loop.create_task(self._force_timeout())
        self._timers = []
        self.setup()

    def setup(self):
        self.setopt(CurlMOpt.TIMERFUNCTION, lib.timer_function)
        self.setopt(CurlMOpt.SOCKETFUNCTION, lib.socket_function)
        self._self_handle = ffi.new_handle(self)
        self.setopt(CurlMOpt.SOCKETDATA, self._self_handle)
        self.setopt(CurlMOpt.TIMERDATA, self._self_handle)

    def close(self):
        self._checker.cancel()
        for curl, future in self._curl2future.items():
            lib.curl_multi_remove_handle(self._curlm, curl._curl)
            future.set_result(None)
        lib.curl_multi_cleanup(self._curlm)
        self._curlm = None
        for sockfd in  self._sockfds:
            self.loop.remove_reader(sockfd)
            self.loop.remove_writer(sockfd)
        for timer in self._timers:
            timer.cancel()

    async def _force_timeout(self):
        while True:
            if not self._curlm:
                break
            await asyncio.sleep(1)
            # print("force timeout")
            self.socket_action(CURL_SOCKET_TIMEOUT, CURL_POLL_NONE)

    async def add_handle(self, curl: Curl, wait=True):
        # import pdb; pdb.set_trace()
        curl.ensure_cacert()
        lib.curl_multi_add_handle(self._curlm, curl._curl)
        future = self.loop.create_future()
        self._curl2future[curl] = future
        self._curl2curl[curl._curl] = curl
        if wait:
            await future

    def socket_action(self, sockfd: int, ev_bitmask: int) -> int:
        running_handle = ffi.new("int *")
        lib.curl_multi_socket_action(self._curlm, sockfd, ev_bitmask, running_handle)
        return running_handle[0]

    def process_data(self, sockfd: int, ev_bitmask: int):
        if not self._curlm:
            print("Curlm alread closed! quitting from process_data")
            return

        self.socket_action(sockfd, ev_bitmask)

        msg_in_queue = ffi.new("int *")
        while True:
            curl_msg = lib.curl_multi_info_read(self._curlm, msg_in_queue)
            # print("message in queue", msg_in_queue[0], curl_msg)
            if curl_msg == ffi.NULL:
                break
            if curl_msg.msg == CURLMSG_DONE:
                # print("curl_message", curl_msg.msg, curl_msg.data.result)
                curl = self._curl2curl[curl_msg.easy_handle]
                retcode = curl_msg.data.result
                if retcode == 0:
                    self.set_result(curl)
                else:
                    # import pdb; pdb.set_trace()
                    self.set_exception(curl, curl._get_error(retcode, "perform"))
            else:
                print("NOT DONE")

    def _pop_future(self, curl: Curl):
        lib.curl_multi_remove_handle(self._curlm, curl._curl)
        self._curl2curl.pop(curl._curl)
        return self._curl2future.pop(curl)

    def cancel_handle(self, curl: Curl):
        future = self._pop_future(curl)
        future.cancel()

    def set_result(self, curl: Curl):
        future = self._pop_future(curl)
        future.set_result(None)

    def set_exception(self, curl: Curl, exception):
        future = self._pop_future(curl)
        future.set_exception(exception)

    def setopt(self, option, value):
        return lib.curl_multi_setopt(self._curlm, option, value)
