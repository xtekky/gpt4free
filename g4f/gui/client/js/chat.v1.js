const colorThemes       = document.querySelectorAll('[name="theme"]');
const markdown          = window.markdownit();
const message_box       = document.getElementById(`messages`);
const message_input     = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const spinner           = box_conversations.querySelector(".spinner");
const stop_generating   = document.querySelector(`.stop_generating`);
const send_button       = document.querySelector(`#send-button`);
let   prompt_lock       = false;

hljs.addPlugin(new CopyButtonPlugin());

const format = (text) => {
    return text.replace(/(?:\r\n|\r|\n)/g, "<br>");
};

message_input.addEventListener("blur", () => {
    window.scrollTo(0, 0);
});

message_input.addEventListener("focus", () => {
    document.documentElement.scrollTop = document.documentElement.scrollHeight;
});

const delete_conversations = async () => {
    localStorage.clear();
    await new_conversation();
};

const handle_ask = async () => {
    message_input.style.height = `80px`;
    message_input.focus();

    let txtMsgs = [];
    const divTags = document.getElementsByClassName("message");
    for(let i=0;i<divTags.length;i++){
        if(!divTags[i].children[1].classList.contains("welcome-message")){
            if(divTags[i].children[0].className == "assistant"){
                const msg = {
                    role: "assistant",
                    content: divTags[i].children[1].textContent+" "
                };
                txtMsgs.push(msg);
            }else{
                const msg = {
                    role: "user",
                    content: divTags[i].children[1].textContent+" "
                };
                txtMsgs.push(msg);
            }
        }
    }

    window.scrollTo(0, 0);
    let message = message_input.value;
    const msg = {
        role: "user",
        content: message
    };
    txtMsgs.push(msg);

    if (message.length > 0) {
        message_input.value = ``;
        await ask_gpt(txtMsgs);
    }
};

const remove_cancel_button = async () => {
    stop_generating.classList.add(`stop_generating-hiding`);

    setTimeout(() => {
        stop_generating.classList.remove(`stop_generating-hiding`);
        stop_generating.classList.add(`stop_generating-hidden`);
    }, 300);
};

const ask_gpt = async (txtMsgs) => {
    try {
        message_input.value     = ``;
        message_input.innerHTML = ``;
        message_input.innerText = ``;

        add_conversation(window.conversation_id, txtMsgs[0].content);
        window.scrollTo(0, 0);
        window.controller = new AbortController();

        jailbreak    = document.getElementById("jailbreak");
        provider     = document.getElementById("provider");
        model        = document.getElementById("model");
        prompt_lock  = true;
        window.text  = ``;
        window.token = message_id();

        stop_generating.classList.remove(`stop_generating-hidden`);

        message_box.innerHTML += `
            <div class="message">
                <div class="user">
                    ${user_image}
                    <i class="fa-regular fa-phone-arrow-up-right"></i>
                </div>
                <div class="content" id="user_${token}"> 
                    ${format(txtMsgs[txtMsgs.length-1].content)}
                </div>
            </div>
        `;

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
                    <div id="cursor"></div>
                </div>
            </div>
        `;

        message_box.scrollTop = message_box.scrollHeight;
        window.scrollTo(0, 0);
        await new Promise((r) => setTimeout(r, 1000));
        window.scrollTo(0, 0);

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
                provider: provider.options[provider.selectedIndex].value,
                meta: {
                    id: window.token,
                    content: {
                        conversation: await get_conversation(window.conversation_id),
                        internet_access: document.getElementById(`switch`).checked,
                        content_type: `text`,
                        parts: txtMsgs,
                    },
                },
            }),
        });

        const reader = response.body.getReader();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            chunk = new TextDecoder().decode(value);

            text += chunk;

            document.getElementById(`gpt_${window.token}`).innerHTML = markdown.render(text).replace("<a href=", '<a target="_blank" href=');
            document.querySelectorAll(`code`).forEach((el) => {
                hljs.highlightElement(el);
            });

            window.scrollTo(0, 0);
            message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
        }

        if (text.includes(`G4F_ERROR`)) {
            document.getElementById(`gpt_${window.token}`).innerHTML = "An error occured, please try again, if the problem persists, please reload / refresh cache or use a differnet browser";
        }

        add_message(window.conversation_id, "user", txtMsgs[txtMsgs.length-1].content);
        add_message(window.conversation_id, "assistant", text);

        message_box.scrollTop = message_box.scrollHeight;
        await remove_cancel_button();
        prompt_lock = false;

        await load_conversations(20, 0);
        window.scrollTo(0, 0);
    
    } catch (e) {
        add_message(window.conversation_id, "user", txtMsgs[txtMsgs.length-1].content);

        message_box.scrollTop = message_box.scrollHeight;
        await remove_cancel_button();
        prompt_lock = false;

        await load_conversations(20, 0);

        console.log(e);

        let cursorDiv = document.getElementById(`cursor`);
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);

        if (e.name != `AbortError`) {
            let error_message = `oops ! something went wrong, please try again / reload. [stacktrace in console]`;

            document.getElementById(`gpt_${window.token}`).innerHTML = error_message;
            add_message(window.conversation_id, "assistant", error_message);
        } else {
            document.getElementById(`gpt_${window.token}`).innerHTML += ` [aborted]`;
            add_message(window.conversation_id, "assistant", text + ` [aborted]`);
        }

        window.scrollTo(0, 0);
    }
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
    let conversation = await JSON.parse(
        localStorage.getItem(`conversation:${conversation_id}`)
    );
    console.log(conversation, conversation_id);

    for (item of conversation.items) {
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
                    ${item.role == "assistant"
                        ? markdown.render(item.content).replace("<a href=", '<a target="_blank" href=')
                        : item.content
                    }
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
};

const add_message = async (conversation_id, role, content) => {
    before_adding = JSON.parse(
        localStorage.getItem(`conversation:${conversation_id}`)
    );

    before_adding.items.push({
        role: role,
        content: content,
    });

    localStorage.setItem(
        `conversation:${conversation_id}`,
        JSON.stringify(before_adding)
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
            <div class="content welcome-message">
            </div>
        </div>
    `;

    content = ``
    to_modify = document.querySelector(`.welcome-message`);
    for (token of tokens) {
        await new Promise(resolve => setTimeout(resolve, (Math.random() * (100 - 200) + 100)))
        content += token;
        to_modify.innerHTML = markdown.render(content);
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

    if (!window.location.href.endsWith(`#`)) {
        if (/\/chat\/.+/.test(window.location.href)) {
            await load_conversation(window.conversation_id);
        }
    }
    
    await say_hello()
    
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