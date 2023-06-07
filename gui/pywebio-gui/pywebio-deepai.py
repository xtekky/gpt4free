import deepai
import time
from pywebio import start_server, config
from pywebio.input import *
from pywebio.output import *
from pywebio.session import local
message_id = ""


def status():
    local.temp_text = ""
    try:
        for chunk in deepai.ChatCompletion.create(local.conversation):
            local.temp_text += chunk
        print(f"Answer: {local.temp_text}")
        put_success(f"Answer: {local.temp_text}", scope="body")
    except:
        put_error("Program Error", scope="body")


def ask(prompt):
    local.conversation.extend([{"role": "user", "content": prompt}])
    local.temp_text = ""
    for chunk in deepai.ChatCompletion.create(local.conversation):
        local.temp_text += chunk
        print(chunk, end="", flush=True)

    local.conversation.extend([
        {"role": "assistant", "content": local.temp_text}
    ])
    return local.temp_text


def msg():
    while True:
        text = input_group(
            "You:", [textarea('You：', name='text', rows=3, placeholder='请输入问题')])
        if not (bool(text)):
            break
        if not (bool(text["text"])):
            continue
        time.sleep(0.5)
        put_code("You："+text["text"], scope="body")
        print("Question："+text["text"])
        with use_scope('foot'):
            put_loading(color="info")
        rp = ask(text["text"])
        clear(scope="foot")
        time.sleep(0.5)
        put_markdown("Bot:\n"+rp, scope="body")
        time.sleep(0.7)


@config(title="AIchat", theme="dark")
def main():
    put_scope("heads")
    with use_scope('heads'):
        put_html("<h1><center>AI Chat</center></h1>")
    put_scope("body")
    put_scope("foot")
    local.temp_text = ""
    local.conversation = [
        {"role": "system", "content": "You are a helpful assistant."}]
    status()
    msg()


print("Click link to chat page")
start_server(main, port=8099, allowed_origins="*",
             auto_open_webbrowser=True, debug=True)
