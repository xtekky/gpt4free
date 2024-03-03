const colorThemes       = document.querySelectorAll('[name="theme"]');
const markdown          = window.markdownit();
const message_box       = document.getElementById(`messages`);
const message_input     = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const stop_generating   = document.querySelector(`.stop_generating`);
const regenerate        = document.querySelector(`.regenerate`);
const sidebar           = document.querySelector(".conversations");
const sidebar_button    = document.querySelector(".mobile-sidebar");
const send_button       = document.getElementById("send-button");
const imageInput        = document.getElementById("image");
const cameraInput       = document.getElementById("camera");
const fileInput         = document.getElementById("file");
const inputCount        = document.getElementById("input-count")
const modelSelect       = document.getElementById("model");

let prompt_lock = false;

hljs.addPlugin(new CopyButtonPlugin());

message_input.addEventListener("blur", () => {
    window.scrollTo(0, 0);
});

message_input.addEventListener("focus", () => {
    document.documentElement.scrollTop = document.documentElement.scrollHeight;
});

const markdown_render = (content) => {
    return markdown.render(content
        .replaceAll(/<!--.+-->/gm, "")
        .replaceAll(/<img data-prompt="[^>]+">/gm, "")
    )
        .replaceAll("<a href=", '<a target="_blank" href=')
        .replaceAll('<code>', '<code class="language-plaintext">')
}

let typesetPromise = Promise.resolve();
const highlight = (container) => {
    container.querySelectorAll('code:not(.hljs').forEach((el) => {
        if (el.className != "hljs") {
            hljs.highlightElement(el);
        }
    });
    typesetPromise = typesetPromise.then(
        () => MathJax.typesetPromise([container])
    ).catch(
        (err) => console.log('Typeset failed: ' + err.message)
    );
}

const register_remove_message = async () => {
    document.querySelectorAll(".message .fa-xmark").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                if (prompt_lock) {
                    return;
                }
                const message_el = el.parentElement.parentElement;
                await remove_message(window.conversation_id, message_el.dataset.index);
                await load_conversation(window.conversation_id);
            })
        }
    });
}

const delete_conversations = async () => {
    localStorage.clear();
    for (let i = 0; i < localStorage.length; i++){
        let key = localStorage.key(i);
        if (key.startsWith("conversation:")) {
            localStorage.removeItem(key);
        }
    }
    hide_sidebar();
    await new_conversation();
};

const handle_ask = async () => {
    message_input.style.height = `82px`;
    message_input.focus();
    window.scrollTo(0, 0);

    message = message_input.value
    if (message.length > 0) {
        message_input.value = '';
        prompt_lock = true;
        count_input()
        await add_conversation(window.conversation_id, message);
        if ("text" in fileInput.dataset) {
            message += '\n```' + fileInput.dataset.type + '\n'; 
            message += fileInput.dataset.text;
            message += '\n```'
        }
        let message_index = await add_message(window.conversation_id, "user", message);
        window.token = message_id();

        if (imageInput.dataset.src) URL.revokeObjectURL(imageInput.dataset.src);
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput
        if (input.files.length > 0) imageInput.dataset.src = URL.createObjectURL(input.files[0]);
        else delete imageInput.dataset.src

        model = modelSelect.options[modelSelect.selectedIndex].value
        message_box.innerHTML += `
            <div class="message" data-index="${message_index}">
                <div class="user">
                    ${user_image}
                    <i class="fa-solid fa-xmark"></i>
                    <i class="fa-regular fa-phone-arrow-up-right"></i>
                </div>
                <div class="content" id="user_${token}"> 
                    <div class="content_inner">
                    ${markdown_render(message)}
                    ${imageInput.dataset.src
                        ? '<img src="' + imageInput.dataset.src + '" alt="Image upload">'
                        : ''
                    }
                    </div>
                    <div class="count">${count_words_and_tokens(message, model)}</div>
                </div>
            </div>
        `;
        await register_remove_message();
        highlight(message_box);
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

const filter_messages = (messages, filter_last_message = true) => {
    // Removes none user messages at end
    if (filter_last_message) {
        let last_message;
        while (last_message = messages.pop()) {
            if (last_message["role"] == "user") {
                messages.push(last_message);
                break;
            }
        }
    }

    // Remove history, if it is selected
    if (document.getElementById('history')?.checked) {
        if (filter_last_message) {
            messages = [messages.pop()];
        } else {
            messages = [messages.pop(), messages.pop()];
        }
    }

    let new_messages = [];
    for (i in messages) {
        new_message = messages[i];
        // Remove generated images from history
        new_message["content"] = new_message["content"].replaceAll(
            /<!-- generated images start -->[\s\S]+<!-- generated images end -->/gm,
            ""
        )
        delete new_message["provider"];
        // Remove regenerated messages
        if (!new_message.regenerate) {
            new_messages.push(new_message)
        }
    }

    return new_messages;
}

const ask_gpt = async () => {
    regenerate.classList.add(`regenerate-hidden`);
    messages = await get_messages(window.conversation_id);
    total_messages = messages.length;

    messages = filter_messages(messages);

    window.scrollTo(0, 0);
    window.controller = new AbortController();

    jailbreak    = document.getElementById("jailbreak");
    provider     = document.getElementById("provider");
    window.text  = '';

    stop_generating.classList.remove(`stop_generating-hidden`);

    message_box.scrollTop = message_box.scrollHeight;
    window.scrollTo(0, 0);
    await new Promise((r) => setTimeout(r, 500));
    window.scrollTo(0, 0);

    el = message_box.querySelector('.count_total');
    el ? el.parentElement.removeChild(el) : null;

    message_box.innerHTML += `
        <div class="message" data-index="${total_messages}">
            <div class="assistant">
                ${gpt_image}
                <i class="fa-solid fa-xmark"></i>
                <i class="fa-regular fa-phone-arrow-down-left"></i>
            </div>
            <div class="content" id="gpt_${window.token}">
                <div class="provider"></div>
                <div class="content_inner"><span id="cursor"></span></div>
                <div class="count"></div>
            </div>
        </div>
    `;
    content = document.getElementById(`gpt_${window.token}`);
    content_inner = content.querySelector('.content_inner');
    content_count = content.querySelector('.count');

    message_box.scrollTop = message_box.scrollHeight;
    window.scrollTo(0, 0);
    try {
        let body = JSON.stringify({
            id: window.token,
            conversation_id: window.conversation_id,
            model: modelSelect.options[modelSelect.selectedIndex].value,
            jailbreak: jailbreak.options[jailbreak.selectedIndex].value,
            web_search: document.getElementById(`switch`).checked,
            provider: provider.options[provider.selectedIndex].value,
            patch_provider: document.getElementById('patch').checked,
            messages: messages
        });
        const headers = {
            accept: 'text/event-stream'
        }
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput
        if (input && input.files.length > 0) {
            const formData = new FormData();
            formData.append('image', input.files[0]);
            formData.append('json', body);
            body = formData;
        } else {
            headers['content-type'] = 'application/json';
        }
        const response = await fetch(`/backend-api/v2/conversation`, {
            method: 'POST',
            signal: window.controller.signal,
            headers: headers,
            body: body
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
                if (message.type == "content") {
                    text += message.content;
                } else if (message["type"] == "provider") {
                    provider = message.provider
                    content.querySelector('.provider').innerHTML = `
                        <a href="${provider.url}" target="_blank">
                            ${provider.name}
                        </a>
                        ${provider.model ? ' with ' + provider.model : ''}
                    `
                } else if (message["type"] == "error") {
                    error = message["error"];
                } else if (message["type"] == "message") {
                    console.error(message["message"])
                }
            }
            if (error) {
                console.error(error);
                content_inner.innerHTML += "<p>An error occured, please try again, if the problem persists, please use a other model or provider.</p>";
            } else {
                html = markdown_render(text);
                let lastElement, lastIndex = null;
                for (element of ['</p>', '</code></pre>', '</li>\n</ol>', '</li>\n</ul>']) {
                    const index = html.lastIndexOf(element)
                    if (index > lastIndex) {
                        lastElement = element;
                        lastIndex = index;
                    }
                }
                if (lastIndex) {
                    html = html.substring(0, lastIndex) + '<span id="cursor"></span>' + lastElement;
                }
                content_inner.innerHTML = html;
                content_count.innerText = count_words_and_tokens(text, provider?.model);
                highlight(content_inner);
            }

            window.scrollTo(0, 0);
            if (message_box.scrollTop >= message_box.scrollHeight - message_box.clientHeight - 100) {
                message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
            }
        }
        if (!error) {
            // Remove cursor
            html = markdown_render(text);
            content_inner.innerHTML = html;
            highlight(content_inner);

            if (imageInput) imageInput.value = "";
            if (cameraInput) cameraInput.value = "";
            if (fileInput) fileInput.value = "";
        }
    } catch (e) {
        console.error(e);

        if (e.name != "AbortError") {
            error = true;
            text = "oops ! something went wrong, please try again / reload. [stacktrace in console]";
            content_inner.innerHTML = text;
        } else {
            content_inner.innerHTML += ` [aborted]`;
            text += ` [aborted]`
        }
    }
    if (!error) {
        await add_message(window.conversation_id, "assistant", text, provider);
        await load_conversation(window.conversation_id);
    } else {
        let cursorDiv = document.getElementById(`cursor`);
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);
    }
    message_box.scrollTop = message_box.scrollHeight;
    await remove_cancel_button();
    await register_remove_message();
    prompt_lock = false;
    window.scrollTo(0, 0);
    await load_conversations();
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

    await load_conversations();
};

const set_conversation = async (conversation_id) => {
    history.pushState({}, null, `/chat/${conversation_id}`);
    window.conversation_id = conversation_id;

    await clear_conversation();
    await load_conversation(conversation_id);
    load_conversations();
    hide_sidebar();
};

const new_conversation = async () => {
    history.pushState({}, null, `/chat/`);
    window.conversation_id = uuid();

    await clear_conversation();
    load_conversations();
    hide_sidebar();
    say_hello();
};

const load_conversation = async (conversation_id) => {
    let messages = await get_messages(conversation_id);

    let elements = "";
    let last_model = null;
    for (i in messages) {
        let item = messages[i];
        last_model = item?.provider?.model;
        let next_i = parseInt(i) + 1;
        let next_provider = item.provider ? item.provider : (messages.length > next_i ? messages[next_i].provider : null);

        let provider_link = item.provider?.name ? `<a href="${item.provider?.url}" target="_blank">${item.provider.name}</a>` : "";
        let provider = provider_link ? `
            <div class="provider">
                ${provider_link}
                ${item.provider.model ? ' with ' + item.provider.model : ''}
            </div>
        ` : "";
        elements += `
            <div class="message" data-index="${i}">
                <div class="${item.role}">
                    ${item.role == "assistant" ? gpt_image : user_image}
                    <i class="fa-solid fa-xmark"></i>
                    ${item.role == "assistant"
                        ? `<i class="fa-regular fa-phone-arrow-down-left"></i>`
                        : `<i class="fa-regular fa-phone-arrow-up-right"></i>`
                    }
                </div>
                <div class="content">
                    ${provider}
                    <div class="content_inner">${markdown_render(item.content)}</div>
                    <div class="count">${count_words_and_tokens(item.content, next_provider?.model)}</div>
                </div>
            </div>
        `;
    }

    const filtered = filter_messages(messages, false);
    if (filtered.length > 0) {
        last_model = last_model?.startsWith("gpt-4") ? "gpt-4" : "gpt-3.5-turbo"
        let count_total = GPTTokenizer_cl100k_base?.encodeChat(filtered, last_model).length
        if (count_total > 0) {
            elements += `<div class="count_total">(${count_total} tokens used)</div>`;
        }
    }

    message_box.innerHTML = elements;

    register_remove_message();
    highlight(message_box);

    message_box.scrollTo({ top: message_box.scrollHeight, behavior: "smooth" });

    setTimeout(() => {
        message_box.scrollTop = message_box.scrollHeight;
    }, 500);
};

// https://stackoverflow.com/questions/20396456/how-to-do-word-counts-for-a-mixture-of-english-and-chinese-in-javascript
function count_words(str) {
    var matches = str.match(/[\u00ff-\uffff]|\S+/g);
    return matches ? matches.length : 0;
}

function count_tokens(model, text) {
    if (model.startsWith("gpt-3") || model.startsWith("gpt-4")) {
        return GPTTokenizer_cl100k_base?.encode(text).length;
    }
    if (model.startsWith("llama2") || model.startsWith("codellama")) {
        return llamaTokenizer?.encode(text).length;
    }
    if (model.startsWith("mistral") || model.startsWith("mixtral")) {
        return mistralTokenizer?.encode(text).length;
    }
}

function count_words_and_tokens(text, model) {
    const tokens_count = model ? count_tokens(model, text) : null;
    const tokens_append = tokens_count ? `, ${tokens_count} tokens` : "";
    return `(${count_words(text)} words${tokens_append})`
}

const get_conversation = async (conversation_id) => {
    let conversation = await JSON.parse(
        localStorage.getItem(`conversation:${conversation_id}`)
    );
    return conversation;
};

const get_messages = async (conversation_id) => {
    let conversation = await get_conversation(conversation_id);
    return conversation?.items || [];
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

const hide_last_message = async (conversation_id) => {
    const conversation = await get_conversation(conversation_id)
    const last_message = conversation.items.pop();
    if (last_message["role"] == "assistant") {
        last_message["regenerate"] = true;
    }
    conversation.items.push(last_message);

    localStorage.setItem(
        `conversation:${conversation_id}`,
        JSON.stringify(conversation)
    );
};

const remove_message = async (conversation_id, index) => {
    const conversation = await get_conversation(conversation_id);
    let new_items = [];
    for (i in conversation.items) {
        if (i == index - 1) {
            delete conversation.items[i]["regenerate"];
        }
        if (i != index) {
            new_items.push(conversation.items[i])
        }
    }
    conversation.items = new_items;
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

    return conversation.items.length - 1;
};

const load_conversations = async () => {
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
};

document.getElementById(`cancelButton`).addEventListener(`click`, async () => {
    window.controller.abort();
    console.log(`aborted ${window.conversation_id}`);
});

document.getElementById(`regenerateButton`).addEventListener(`click`, async () => {
    prompt_lock = true;
    await hide_last_message(window.conversation_id);
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

async function hide_sidebar() {
    sidebar.classList.remove("shown");
    sidebar_button.classList.remove("rotated");
}

sidebar_button.addEventListener("click", (event) => {
    if (sidebar.classList.contains("shown")) {
        hide_sidebar();
    } else {
        sidebar.classList.add("shown");
        sidebar_button.classList.add("rotated");
    }

    window.scrollTo(0, 0);
});

const register_settings_localstorage = async () => {
    for (id of ["switch", "model", "jailbreak", "patch", "provider", "history"]) {
        element = document.getElementById(id);
        element.addEventListener('change', async (event) => {
            switch (event.target.type) {
                case "checkbox":
                    localStorage.setItem(id, event.target.checked);
                    break;
                case "select-one":
                    localStorage.setItem(id, event.target.selectedIndex);
                    break;
                default:
                    console.warn("Unresolved element type");
            }
        });
    }
}

const load_settings_localstorage = async () => {
    for (id of ["switch", "model", "jailbreak", "patch", "provider", "history"]) {
        element = document.getElementById(id);
        value = localStorage.getItem(element.id);
        if (value) {
            switch (element.type) {
                case "checkbox":
                    element.checked = value === "true";
                    break;
                case "select-one":
                    element.selectedIndex = parseInt(value);
                    break;
                default:
                    console.warn("Unresolved element type");
            }
        }
    }
}

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

const count_input = async () => {
    if (message_input.value) {
        model = modelSelect.options[modelSelect.selectedIndex].value;
        inputCount.innerText = count_words_and_tokens(message_input.value, model);
    } else {
        inputCount.innerHTML = "&nbsp;"
    }
};
message_input.addEventListener("keyup", count_input);

window.onload = async () => {
    setTheme();

    count_input();

    if (/\/chat\/.+/.test(window.location.href)) {
        load_conversation(window.conversation_id);
    } else {
        say_hello()
    }

    setTimeout(() => {
        load_conversations();
    }, 1);

    message_input.addEventListener("keydown", async (evt) => {
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

    for (model of models) {
        let option = document.createElement('option');
        option.value = option.text = model;
        modelSelect.appendChild(option);
    }

    response = await fetch('/backend-api/v2/providers')
    providers = await response.json()
    select = document.getElementById('provider');

    for (provider of providers) {
        let option = document.createElement('option');
        option.value = option.text = provider;
        select.appendChild(option);
    }

    await load_settings_localstorage()
})();

(async () => {
    response = await fetch('/backend-api/v2/version')
    versions = await response.json()
    
    document.title = 'g4f - gui - ' + versions["version"];
    let text = "version ~ "
    if (versions["version"] != versions["latest_version"]) {
        let release_url = 'https://github.com/xtekky/gpt4free/releases/tag/' + versions["latest_version"];
        let title = `New version: ${versions["latest_version"]}`;
        text += `<a href="${release_url}" target="_blank" title="${title}">${versions["version"]} 🆕</a>`;
    } else {
        text += versions["version"];
    }
    document.getElementById("version_text").innerHTML = text
})()

for (const el of [imageInput, cameraInput]) {
    el.addEventListener('click', async () => {
        el.value = '';
        if (imageInput.dataset.src) {
            URL.revokeObjectURL(imageInput.dataset.src);
            delete imageInput.dataset.src
        }
    });
}

fileInput.addEventListener('click', async (event) => {
    fileInput.value = '';
    delete fileInput.dataset.text;
});
fileInput.addEventListener('change', async (event) => {
    if (fileInput.files.length) {
        type = fileInput.files[0].type;
        if (type && type.indexOf('/')) {
            type = type.split('/').pop().replace('x-', '')
            type = type.replace('plain', 'plaintext')
                       .replace('shellscript', 'sh')
                       .replace('svg+xml', 'svg')
                       .replace('vnd.trolltech.linguist', 'ts')
        } else {
            type = fileInput.files[0].name.split('.').pop()
        }
        fileInput.dataset.type = type
        const reader = new FileReader();
        reader.addEventListener('load', (event) => {
            fileInput.dataset.text = event.target.result;
        });
        reader.readAsText(fileInput.files[0]);
    } else {
        delete fileInput.dataset.text;
    }
});