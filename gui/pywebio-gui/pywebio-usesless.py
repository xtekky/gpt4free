import bleach
import time
from gpt4free import usesless
from pywebio import start_server, config
from pywebio.input import *
from pywebio.output import *
from pywebio.session import local

def sanitize_text(text):
    """
    Sanitize user input to prevent code injection attacks
    """
    return bleach.clean(text)

def ask(prompt):
    """
    Generate a response from the AI model
    """
    req = usesless.Completion.create(prompt=prompt, parentMessageId=local.message_id)
    rp = sanitize_text(req['text'])
    local.message_id = req["id"]
    print("AI:\n" + rp)
    local.conversation.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rp}
    ])
    print(local.conversation)
    return rp

def msg():
    """
    Handle incoming messages and generate responses
    """
    while True:
        text = input_group("You:", [textarea('You:', name='text', rows=3, placeholder='Enter your question')])
        if not bool(text):
            break
        if not bool(text["text"]):
            continue
        time.sleep(0.5)
        put_code("You: " + text["text"], scope="body")
        print("Question: " + text["text"])
        with use_scope('foot'):
            put_loading(color="info")
        rp = ask(text["text"])
        clear(scope="foot")
        time.sleep(0.5)
        put_text("Bot:\n" + rp, scope="body")
        time.sleep(0.7)

@config(title="AI chat", theme="dark")
def main():
    """
    Start the web server and run the chatbot
    """
    put_scope("heads")
    with use_scope('heads'):
        put_html("<h1><center>AI Chat</center></h1>")
    put_scope("body")
    put_scope("foot")
    # Generate initial status message from AI model
    try:
        req = usesless.Completion.create(prompt="hello", parentMessageId=local.message_id)
        status_text = sanitize_text(req['text'])
        put_success(f"Answer: {status_text}", scope="body")
    except:
        put_error("Program Error", scope="body")
    # Initialize conversation variables
    local.conversation = []
    local.message_id = ""
    # Start the chat loop
    msg()

print("Click link to chat page")
start_server(main, port=8099, allowed_origins="*", auto_open_webbrowser=True, debug=True)
