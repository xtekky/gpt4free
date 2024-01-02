const colorThemes       = document.querySelectorAll('[name="theme"]');
const markdown          = window.markdownit();
const message_box       = document.getElementById(`messages`);
const message_input     = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const spinner           = box_conversations.querySelector(".spinner");
const stop_generating   = document.querySelector(`.stop_generating`);
const regenerate        = document.querySelector(`.regenerate`);
const send_button       = document.querySelector(`#send-button`);
let   prompt_lock       = false;

hljs.addPlugin(new CopyButtonPlugin());

message_input.addEventListener("blur", () => {
    window.scrollTo(0, 0);
});

message_input.addEventListener("focus", () => {
    document.documentElement.scrollTop = document.documentElement.scrollHeight;
});

const markdown_render = (content) => {
    return markdown.render(content)
        .replaceAll("<a href=", '<a target="_blank" href=')
        .replaceAll('<code>', '<code class="language-plaintext">')
}

const delete_conversations = async () => {
    localStorage.clear();
    await new_conversation();
};

const handle_ask = async () => {
    message_input.style.height = `80px`;
    message_input.focus();
    window.scrollTo(0, 0);
    message = message_input.value
    if (message.length > 0) {
        message_input.value = '';
        await add_conversation(window.conversation_id, message);
        await add_message(window.conversation_id, "user", message);
        window.token = message_id();
        message_box.innerHTML += `
            <div class="message">
                <div class="user">
                    ${user_image}
                    <i class="fa-regular fa-phone-arrow-up-right"></i>
                </div>
                <div class="content" id="user_${token}"> 
                    ${markdown_render(message)}
                </div>
            </div>
        `;
        await ask_gpt();
    }
};

const remove_cancel_button = async () => {
    stop_generating.classList.add(`stop_generating-hiding`);

    setTimeout(() => {
        stop_generating.classList.remove(`stop_generating-hiding`);
        stop_generating.classList.add(`stop_generating-hidden`);
    }, 300);
};

const ask_gpt = async () => {
    regenerate.classList.add(`regenerate-hidden`);
    messages = await get_messages(window.conversation_id);

    window.scrollTo(0, 0);
    window.controller = new AbortController();

    jailbreak    = document.getElementById("jailbreak");
    provider     = document.getElementById("provider");
    model        = document.getElementById("model");
    prompt_lock  = true;
    window.text  = '';

    stop_generating.classList.remove(`stop_generating-hidden`);

    message_box.scrollTop = message_box.scrollHeight;
    window.scrollTo(0, 0);
    await new Promise((r) => setTimeout(r, 500));
    window.scrollTo(0, 0);

    message_box.innerHTML += `
        <div class="message">
            <div class="assistant">
                ${gpt_image} <i class="fa-regular fa-phone-arrow-down-left"></i>
            </div>
            <div class="content" id="gpt_${window.token}">
                <div class="provider"></div>
                <div class="content_inner"><div id="cursor"></div></div>
            </div>
        </div>
    `;
    content = document.getElementById(`gpt_${window.token}`);
    content_inner = content.querySelector('.content_inner');

    message_box.scrollTop = message_box.scrollHeight;
    window.scrollTo(0, 0);
    try {
        const response = await fetch(`/backend-api/v2/conversation`, {
            method: `POST`,
            signal: window.controller.signal,
            headers: {
                'content-type': `application/json`,
                accept: `text/event-stream`,
            },
            body: JSON.stringify({
                conversation_id: window.conversation_id,
                action: `_ask`,
                model: model.options[model.selectedIndex].value,
                jailbreak: jailbreak.options[jailbreak.selectedIndex].value,
                internet_access: document.getElementById(`switch`).checked,
                provider: provider.options[provider.selectedIndex].value,
                meta: {
                    id: window.token,
                    content: {
                        content_type: `text`,
                        parts: messages,
                    },
                },
            }),
        });

        await new Promise((r) => setTimeout(r, 1000));
        window.scrollTo(0, 0);

        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

        error = provider = null;
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            for (const line of value.split("\n")) {
                if (!line) continue;
                const message = JSON.parse(line);
                if (message["type"] == "content") {
                    text += message["content"];
                } else if (message["type"] == "provider") {
                    provider = message["provider"];
                    content.querySelector('.provider').innerHTML =
                        '<a href="' + provider.url + '" target="_blank">' + provider.name + "</a>"
                } else if (message["type"] == "error") {
                    error = message["error"];
                }
            }
            if (error) {
                console.error(error);
                content_inner.innerHTML = "An error occured, please try again, if the problem persists, please use a other model or provider";
            } else {
                content_inner.innerHTML = markdown_render(text);
                document.querySelectorAll('code').forEach((el) => {
                    hljs.highlightElement(el);
                });
            }

            window.scrollTo(0, 0);
            message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
        }
    } catch (e) {
        console.log(e);

        let cursorDiv = document.getElementById(`cursor`);
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);

        if (e.name != `AbortError`) {
            text = `oops ! something went wrong, please try again / reload. [stacktrace in console]`;
            content_inner.innerHTML = text;
        } else {
            content_inner.innerHTML += ` [aborted]`;
            text += ` [aborted]`
        }
    }
    add_message(window.conversation_id, "assistant", text, provider);
    message_box.scrollTop = message_box.scrollHeight;
    await remove_cancel_button();
    prompt_lock = false;
    window.scrollTo(0, 0);
    await load_conversations(20, 0);
    regenerate.classList.remove(`regenerate-hidden`);
};

const clear_conversations = async () => {
    const elements = box_conversations.childNodes;
    let index = elements.length;

    if (index > 0) {
        while (index--) {
            const element = elements[index];
            if (
                element.nodeType === Node.ELEMENT_NODE &&
                element.tagName.toLowerCase() !== `button`
            ) {
                box_conversations.removeChild(element);
            }
        }
    }
};

const clear_conversation = async () => {
    let messages = message_box.getElementsByTagName(`div`);

    while (messages.length > 0) {
        message_box.removeChild(messages[0]);
    }
};

const show_option = async (conversation_id) => {
    const conv = document.getElementById(`conv-${conversation_id}`);
    const yes = document.getElementById(`yes-${conversation_id}`);
    const not = document.getElementById(`not-${conversation_id}`);

    conv.style.display = `none`;
    yes.style.display  = `block`;
    not.style.display  = `block`;
};

const hide_option = async (conversation_id) => {
    const conv = document.getElementById(`conv-${conversation_id}`);
    const yes  = document.getElementById(`yes-${conversation_id}`);
    const not  = document.getElementById(`not-${conversation_id}`);

    conv.style.display = `block`;
    yes.style.display  = `none`;
    not.style.display  = `none`;
};

const delete_conversation = async (conversation_id) => {
    localStorage.removeItem(`conversation:${conversation_id}`);

    const conversation = document.getElementById(`convo-${conversation_id}`);
    conversation.remove();

    if (window.conversation_id == conversation_id) {
        await new_conversation();
    }

    await load_conversations(20, 0, true);
};

const set_conversation = async (conversation_id) => {
    history.pushState({}, null, `/chat/${conversation_id}`);
    window.conversation_id = conversation_id;

    await clear_conversation();
    await load_conversation(conversation_id);
    await load_conversations(20, 0, true);
};

const new_conversation = async () => {
    history.pushState({}, null, `/chat/`);
    window.conversation_id = uuid();

    await clear_conversation();
    await load_conversations(20, 0, true);

    await say_hello()
};

const load_conversation = async (conversation_id) => {
    let messages = await get_messages(conversation_id);

    for (item of messages) {
        message_box.innerHTML += `
            <div class="message">
                <div class=${item.role == "assistant" ? "assistant" : "user"}>
                    ${item.role == "assistant" ? gpt_image : user_image}
                    ${item.role == "assistant"
                        ? `<i class="fa-regular fa-phone-arrow-down-left"></i>`
                        : `<i class="fa-regular fa-phone-arrow-up-right"></i>`
                    }
                </div>
                <div class="content">
                    ${item.provider
                        ? '<div class="provider"><a href="' + item.provider.url + '" target="_blank">' + item.provider.name + '</a></div>'
                        : ''
                    }
                    <div class="content_inner">${markdown_render(item.content)}</div>
                </div>
            </div>
        `;
    }

    document.querySelectorAll(`code`).forEach((el) => {
        hljs.highlightElement(el);
    });

    message_box.scrollTo({ top: message_box.scrollHeight, behavior: "smooth" });

    setTimeout(() => {
        message_box.scrollTop = message_box.scrollHeight;
    }, 500);
};

const get_conversation = async (conversation_id) => {
    let conversation = await JSON.parse(
        localStorage.getItem(`conversation:${conversation_id}`)
    );
    return conversation;
};

const get_messages = async (conversation_id) => {
    let conversation = await get_conversation(conversation_id);
    return conversation.items;
};

const add_conversation = async (conversation_id, content) => {
    if (content.length > 17) {
        title = content.substring(0, 17) + '...'
    } else {
        title = content + '&nbsp;'.repeat(19 - content.length)
    }

    if (localStorage.getItem(`conversation:${conversation_id}`) == null) {
        localStorage.setItem(
            `conversation:${conversation_id}`,
            JSON.stringify({
                id: conversation_id,
                title: title,
                items: [],
            })
        );
    }

    history.pushState({}, null, `/chat/${conversation_id}`);
};

const remove_last_message = async (conversation_id) => {
    const conversation = await get_conversation(conversation_id)

    conversation.items.pop();

    localStorage.setItem(
        `conversation:${conversation_id}`,
        JSON.stringify(conversation)
    );
};

const add_message = async (conversation_id, role, content, provider) => {
    const conversation = await get_conversation(conversation_id);

    conversation.items.push({
        role: role,
        content: content,
        provider: provider
    });

    localStorage.setItem(
        `conversation:${conversation_id}`,
        JSON.stringify(conversation)
    );
};

const load_conversations = async (limit, offset, loader) => {
    let conversations = [];
    for (let i = 0; i < localStorage.length; i++) {
        if (localStorage.key(i).startsWith("conversation:")) {
            let conversation = localStorage.getItem(localStorage.key(i));
            conversations.push(JSON.parse(conversation));
        }
    }

    await clear_conversations();

    for (conversation of conversations) {
        box_conversations.innerHTML += `
            <div class="convo" id="convo-${conversation.id}">
                <div class="left" onclick="set_conversation('${conversation.id}')">
                    <i class="fa-regular fa-comments"></i>
                    <span class="convo-title">${conversation.title}</span>
                </div>
                <i onclick="show_option('${conversation.id}')" class="fa-regular fa-trash" id="conv-${conversation.id}"></i>
                <i onclick="delete_conversation('${conversation.id}')" class="fa-regular fa-check" id="yes-${conversation.id}" style="display:none;"></i>
                <i onclick="hide_option('${conversation.id}')" class="fa-regular fa-x" id="not-${conversation.id}" style="display:none;"></i>
            </div>
        `;
    }

    document.querySelectorAll(`code`).forEach((el) => {
        hljs.highlightElement(el);
    });
};

document.getElementById(`cancelButton`).addEventListener(`click`, async () => {
    window.controller.abort();
    console.log(`aborted ${window.conversation_id}`);
});

document.getElementById(`regenerateButton`).addEventListener(`click`, async () => {
    await remove_last_message(window.conversation_id);
    window.token = message_id();
    await ask_gpt();
});

const uuid = () => {
    return `xxxxxxxx-xxxx-4xxx-yxxx-${Date.now().toString(16)}`.replace(
        /[xy]/g,
        function (c) {
            var r = (Math.random() * 16) | 0,
                v = c == "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        }
    );
};

const message_id = () => {
    random_bytes = (Math.floor(Math.random() * 1338377565) + 2956589730).toString(
        2
    );
    unix = Math.floor(Date.now() / 1000).toString(2);

    return BigInt(`0b${unix}${random_bytes}`).toString();
};

document.querySelector(".mobile-sidebar").addEventListener("click", (event) => {
    const sidebar = document.querySelector(".conversations");

    if (sidebar.classList.contains("shown")) {
        sidebar.classList.remove("shown");
        event.target.classList.remove("rotated");
    } else {
        sidebar.classList.add("shown");
        event.target.classList.add("rotated");
    }

    window.scrollTo(0, 0);
});

const register_settings_localstorage = async () => {
    settings_ids = ["switch", "model", "jailbreak"];
    settings_elements = settings_ids.map((id) => document.getElementById(id));
    settings_elements.map((element) =>
        element.addEventListener(`change`, async (event) => {
            switch (event.target.type) {
                case "checkbox":
                    localStorage.setItem(event.target.id, event.target.checked);
                    break;
                case "select-one":
                    localStorage.setItem(event.target.id, event.target.selectedIndex);
                    break;
                default:
                    console.warn("Unresolved element type");
            }
        })
    );
};

const load_settings_localstorage = async () => {
    settings_ids = ["switch", "model", "jailbreak"];
    settings_elements = settings_ids.map((id) => document.getElementById(id));
    settings_elements.map((element) => {
        if (localStorage.getItem(element.id)) {
            switch (element.type) {
                case "checkbox":
                    element.checked = localStorage.getItem(element.id) === "true";
                    break;
                case "select-one":
                    element.selectedIndex = parseInt(localStorage.getItem(element.id));
                    break;
                default:
                    console.warn("Unresolved element type");
            }
        }
    });
};

const say_hello = async () => {
    tokens = [`Hello`, `!`, ` How`,` can`, ` I`,` assist`,` you`,` today`,`?`]

    message_box.innerHTML += `
        <div class="message">
            <div class="assistant">
                ${gpt_image}
                <i class="fa-regular fa-phone-arrow-down-left"></i>
            </div>
            <div class="content">
                <p class=" welcome-message"></p>
            </div>
        </div>
    `;

    to_modify = document.querySelector(`.welcome-message`);
    for (token of tokens) {
        await new Promise(resolve => setTimeout(resolve, (Math.random() * (100 - 200) + 100)))
        to_modify.textContent += token;
    }
}

// Theme storage for recurring viewers
const storeTheme = function (theme) {
    localStorage.setItem("theme", theme);
};

// set theme when visitor returns
const setTheme = function () {
    const activeTheme = localStorage.getItem("theme");
    colorThemes.forEach((themeOption) => {
        if (themeOption.id === activeTheme) {
            themeOption.checked = true;
        }
    });
    // fallback for no :has() support
    document.documentElement.className = activeTheme;
};

colorThemes.forEach((themeOption) => {
    themeOption.addEventListener("click", () => {
        storeTheme(themeOption.id);
        // fallback for no :has() support
        document.documentElement.className = themeOption.id;
    });
});


window.onload = async () => {
    load_settings_localstorage();
    setTheme();

    let conversations = 0;
    for (let i = 0; i < localStorage.length; i++) {
        if (localStorage.key(i).startsWith("conversation:")) {
            conversations += 1;
        }
    }

    if (conversations == 0) localStorage.clear();

    await setTimeout(() => {
        load_conversations(20, 0);
    }, 1);

    if (/\/chat\/.+/.test(window.location.href)) {
        await load_conversation(window.conversation_id);
    } else {
        await say_hello()
    }
        
    message_input.addEventListener(`keydown`, async (evt) => {
        if (prompt_lock) return;
        if (evt.keyCode === 13 && !evt.shiftKey) {
            evt.preventDefault();
            console.log("pressed enter");
            await handle_ask();
        } else {
            message_input.style.removeProperty("height");
            message_input.style.height = message_input.scrollHeight  + "px";
        }
    });
    
    send_button.addEventListener(`click`, async () => {
        console.log("clicked send");
        if (prompt_lock) return;
        await handle_ask();
    });

    register_settings_localstorage();
};

const observer = new MutationObserver((mutationsList) => {
    for (const mutation of mutationsList) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
            const height = message_input.offsetHeight;
            
            let heightValues = {
                81: "20px",
                82: "20px",
                100: "30px",
                119: "39px",
                138: "49px",
                150: "55px"
            }
            
            send_button.style.top = heightValues[height] || '';
        }
    }
});

observer.observe(message_input, { attributes: true });

(async () => {
    response = await fetch('/backend-api/v2/models')
    models = await response.json()
    
    let select = document.getElementById('model');
    select.textContent = '';

    let auto = document.createElement('option');
    auto.value = '';
    auto.text = 'Model: Default';
    select.appendChild(auto);

    for (model of models) {
        let option = document.createElement('option');
        option.value = option.text = model;
        select.appendChild(option);
    }
})();

(async () => {
    response = await fetch('/backend-api/v2/providers')
    providers = await response.json()
    
    let select = document.getElementById('provider');
    select.textContent = '';

    let auto = document.createElement('option');
    auto.value = '';
    auto.text = 'Provider: Auto';
    select.appendChild(auto);

    for (provider of providers) {
        let option = document.createElement('option');
        option.value = option.text = provider;
        select.appendChild(option);
    }
})();

(async () => {
    response = await fetch('/backend-api/v2/version')
    versions = await response.json()
    
    document.title = 'g4f - gui - ' + versions["version"];
    text = "version ~ "
    if (versions["version"] != versions["lastet_version"]) {
        release_url = 'https://github.com/xtekky/gpt4free/releases/tag/' + versions["lastet_version"];
        text += '<a href="' + release_url +'" target="_blank" title="New version: ' + versions["lastet_version"] +'">' + versions["version"] + ' 🆕</a>';
    } else {
        text += versions["version"];
    }
    document.getElementById("version_text").innerHTML = text
})();