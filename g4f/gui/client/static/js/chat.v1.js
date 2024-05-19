const colorThemes       = document.querySelectorAll('[name="theme"]');
const message_box       = document.getElementById(`messages`);
const messageInput      = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const stop_generating   = document.querySelector(`.stop_generating`);
const regenerate        = document.querySelector(`.regenerate`);
const sidebar           = document.querySelector(".conversations");
const sidebar_button    = document.querySelector(".mobile-sidebar");
const sendButton        = document.getElementById("send-button");
const imageInput        = document.getElementById("image");
const cameraInput       = document.getElementById("camera");
const fileInput         = document.getElementById("file");
const microLabel        = document.querySelector(".micro-label");
const inputCount        = document.getElementById("input-count").querySelector(".text");
const providerSelect    = document.getElementById("provider");
const modelSelect       = document.getElementById("model");
const modelProvider     = document.getElementById("model2");
const systemPrompt      = document.getElementById("systemPrompt");
const settings          = document.querySelector(".settings");
const chat              = document.querySelector(".conversation");
const album             = document.querySelector(".images");

let prompt_lock = false;

let content, content_inner, content_count = null;

const optionElements = document.querySelectorAll(".settings input, .settings textarea, #model, #model2, #provider")

messageInput.addEventListener("blur", () => {
    window.scrollTo(0, 0);
});

messageInput.addEventListener("focus", () => {
    document.documentElement.scrollTop = document.documentElement.scrollHeight;
});

appStorage = window.localStorage || {
    setItem: (key, value) => self[key] = value,
    getItem: (key) => self[key],
    removeItem: (key) => delete self[key],
    length: 0
}

const markdown = window.markdownit();
const markdown_render = (content) => {
    return markdown.render(content
        .replaceAll(/<!-- generated images start -->|<!-- generated images end -->/gm, "")
        .replaceAll(/<img data-prompt="[^>]+">/gm, "")
    )
        .replaceAll("<a href=", '<a target="_blank" href=')
        .replaceAll('<code>', '<code class="language-plaintext">')
}

function filter_message(text) {
    return text.replaceAll(
        /<!-- generated images start -->[\s\S]+<!-- generated images end -->/gm, ""
    )
}

hljs.addPlugin(new CopyButtonPlugin());
let typesetPromise = Promise.resolve();
const highlight = (container) => {
    container.querySelectorAll('code:not(.hljs').forEach((el) => {
        if (el.className != "hljs") {
            hljs.highlightElement(el);
        }
    });
    if (window.MathJax) {
        typesetPromise = typesetPromise.then(
            () => MathJax.typesetPromise([container])
        ).catch(
            (err) => console.log('Typeset failed: ' + err.message)
        );
    }
}

const register_message_buttons = async () => {
    document.querySelectorAll(".message .fa-xmark").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                if (prompt_lock) {
                    return;
                }
                const message_el = el.parentElement.parentElement;
                await remove_message(window.conversation_id, message_el.dataset.index);
                await load_conversation(window.conversation_id, false);
            })
        }
    });
    document.querySelectorAll(".message .fa-clipboard").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = el.parentElement.parentElement.parentElement;
                const copyText = await get_message(window.conversation_id, message_el.dataset.index);
                navigator.clipboard.writeText(copyText);
                el.classList.add("clicked");
                setTimeout(() => el.classList.remove("clicked"), 1000);
            })
        }
    });
    document.querySelectorAll(".message .fa-volume-high").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                let playlist = [];
                function play_next() {
                    const next = playlist.shift();
                    if (next && el.dataset.do_play) {
                        next.play();
                    }
                }
                if (el.dataset.stopped) {
                    el.classList.remove("blink")
                    delete el.dataset.stopped;
                    return;
                }
                if (el.dataset.running) {
                    el.dataset.stopped = true;
                    el.classList.add("blink")
                    playlist = [];
                    return;
                }
                el.dataset.running = true;
                el.classList.add("blink")
                el.classList.add("active")
                const content_el = el.parentElement.parentElement;
                const message_el = content_el.parentElement;
                let speechText = await get_message(window.conversation_id, message_el.dataset.index);

                speechText = speechText.replaceAll(/([^0-9])\./gm, "$1.;");
                speechText = speechText.replaceAll("?", "?;");
                speechText = speechText.replaceAll(/\[(.+)\]\(.+\)/gm, "($1)");
                speechText = speechText.replaceAll(/```[a-z]+/gm, "");
                speechText = filter_message(speechText.replaceAll("`", "").replaceAll("#", ""))
                const lines = speechText.trim().split(/\n|;/).filter(v => count_words(v));

                window.onSpeechResponse = (url) => {
                    if (!el.dataset.stopped) {
                        el.classList.remove("blink")
                    }
                    if (url) {
                        var sound = document.createElement('audio');
                        sound.controls = 'controls';
                        sound.src = url;
                        sound.type = 'audio/wav';
                        sound.onended = function() {
                            el.dataset.do_play = true;
                            setTimeout(play_next, 1000);
                        };
                        sound.onplay = function() {
                            delete el.dataset.do_play;
                        };
                        var container = document.createElement('div');
                        container.classList.add("audio");
                        container.appendChild(sound);
                        content_el.appendChild(container);
                        if (!el.dataset.stopped) {
                            playlist.push(sound);
                            if (el.dataset.do_play) {
                                play_next();
                            }
                        }
                    }
                    let line = lines.length > 0 ? lines.shift() : null;
                    if (line && !el.dataset.stopped) {
                        handleGenerateSpeech(line);
                    } else {
                        el.classList.remove("active");
                        el.classList.remove("blink");
                        delete el.dataset.running;
                    }
                }
                el.dataset.do_play = true;
                let line = lines.shift();
                handleGenerateSpeech(line);
            });
        }
    });
    document.querySelectorAll(".message .fa-rotate").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = el.parentElement.parentElement.parentElement;
                el.classList.add("clicked");
                setTimeout(() => el.classList.remove("clicked"), 1000);
                prompt_lock = true;
                await hide_message(window.conversation_id, message_el.dataset.index);
                window.token = message_id();
                await ask_gpt(message_el.dataset.index);
            })
        }
    });
    document.querySelectorAll(".message .fa-whatsapp").forEach(async (el) => {
        if (!el.parentElement.href) {
            const text = el.parentElement.parentElement.parentElement.innerText;
            el.parentElement.href = `https://wa.me/?text=${encodeURIComponent(text)}`;
        }
    });
    document.querySelectorAll(".message .fa-print").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = el.parentElement.parentElement.parentElement;
                el.classList.add("clicked");
                message_box.scrollTop = 0;
                message_el.classList.add("print");
                setTimeout(() => el.classList.remove("clicked"), 1000);
                setTimeout(() => message_el.classList.remove("print"), 1000);
                window.print()
            })
        }
    });
}

const delete_conversations = async () => {
    const remove_keys = [];
    for (let i = 0; i < appStorage.length; i++){
        let key = appStorage.key(i);
        if (key.startsWith("conversation:")) {
            remove_keys.push(key);
        }
    }
    remove_keys.forEach((key)=>appStorage.removeItem(key));
    hide_sidebar();
    await new_conversation();
};

const handle_ask = async () => {
    messageInput.style.height = "82px";
    messageInput.focus();
    window.scrollTo(0, 0);

    message = messageInput.value
    if (message.length <= 0) {
        return;
    }
    messageInput.value = "";
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
                <div class="count">
                    ${count_words_and_tokens(message, get_selected_model())}
                    <i class="fa-solid fa-volume-high"></i>
                    <i class="fa-regular fa-clipboard"></i>
                    <a><i class="fa-brands fa-whatsapp"></i></a>
                    <i class="fa-solid fa-print"></i>
                </div>
            </div>
        </div>
    `;
    highlight(message_box);
    await ask_gpt();
};

const remove_cancel_button = async () => {
    stop_generating.classList.add(`stop_generating-hiding`);

    setTimeout(() => {
        stop_generating.classList.remove(`stop_generating-hiding`);
        stop_generating.classList.add(`stop_generating-hidden`);
    }, 300);
};

const prepare_messages = (messages, message_index = -1) => {
    // Removes none user messages at end
    if (message_index == -1) {
        let last_message;
        while (last_message = messages.pop()) {
            if (last_message["role"] == "user") {
                messages.push(last_message);
                break;
            }
        }
    } else if (message_index >= 0) {
        messages = messages.filter((_, index) => message_index >= index);
    }

    // Remove history, if it's selected
    if (document.getElementById('history')?.checked) {
        if (message_index == null) {
            messages = [messages.pop(), messages.pop()];
        } else {
            messages = [messages.pop()];
        }
    }

    let new_messages = [];
    if (systemPrompt?.value) {
        new_messages.push({
            "role": "system",
            "content": systemPrompt.value
        });
    }
    messages.forEach((new_message) => {
        // Include only not regenerated messages
        if (new_message && !new_message.regenerate) {
            // Remove generated images from history
            new_message.content = filter_message(new_message.content);
            delete new_message.provider;
            new_messages.push(new_message)
        }
    });
    return new_messages;
}

async function add_message_chunk(message) {
    if (message.type == "conversation") {
        console.info("Conversation used:", message.conversation)
    } else if (message.type == "provider") {
        window.provider_result = message.provider;
        content.querySelector('.provider').innerHTML = `
            <a href="${message.provider.url}" target="_blank">
                ${message.provider.label ? message.provider.label : message.provider.name}
            </a>
            ${message.provider.model ? ' with ' + message.provider.model : ''}
        `
    } else if (message.type == "message") {
        console.error(message.message)
    } else if (message.type == "error") {
        window.error = message.error
        console.error(message.error);
        content_inner.innerHTML += `<p><strong>An error occured:</strong> ${message.error}</p>`;
    } else if (message.type == "preview") {
        content_inner.innerHTML = markdown_render(message.preview);
    } else if (message.type == "content") {
        window.text += message.content;
        html = markdown_render(window.text);
        let lastElement, lastIndex = null;
        for (element of ['</p>', '</code></pre>', '</p>\n</li>\n</ol>', '</li>\n</ol>', '</li>\n</ul>']) {
            const index = html.lastIndexOf(element)
            if (index - element.length > lastIndex) {
                lastElement = element;
                lastIndex = index;
            }
        }
        if (lastIndex) {
            html = html.substring(0, lastIndex) + '<span id="cursor"></span>' + lastElement;
        }
        content_inner.innerHTML = html;
        content_count.innerText = count_words_and_tokens(text, window.provider_result?.model);
        highlight(content_inner);
    }
    window.scrollTo(0, 0);
    if (message_box.scrollTop >= message_box.scrollHeight - message_box.clientHeight - 100) {
        message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
    }
}

// fileInput?.addEventListener("click", (e) => {
//     if (window?.pywebview) {
//         e.preventDefault();
//         pywebview.api.choose_file();
//     }
// });

cameraInput?.addEventListener("click", (e) => {
    if (window?.pywebview) {
        e.preventDefault();
        pywebview.api.take_picture();
    }
});

imageInput?.addEventListener("click", (e) => {
    if (window?.pywebview) {
        e.preventDefault();
        pywebview.api.choose_image();
    }
});

const ask_gpt = async (message_index = -1) => {
    regenerate.classList.add(`regenerate-hidden`);
    messages = await get_messages(window.conversation_id);
    total_messages = messages.length;
    messages = prepare_messages(messages, message_index);

    stop_generating.classList.remove(`stop_generating-hidden`);

    message_box.scrollTop = message_box.scrollHeight;
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

    window.controller = new AbortController();
    window.text  = "";
    window.error = null;
    window.abort = false;
    window.provider_result = null;

    content = document.getElementById(`gpt_${window.token}`);
    content_inner = content.querySelector('.content_inner');
    content_count = content.querySelector('.count');

    message_box.scrollTop = message_box.scrollHeight;
    window.scrollTo(0, 0);
    try {
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput;
        const file = input && input.files.length > 0 ? input.files[0] : null;
        const provider = providerSelect.options[providerSelect.selectedIndex].value;
        const auto_continue = document.getElementById("auto_continue")?.checked;
        if (file && !provider)
            provider = "Bing";
        let api_key = null;
        if (provider) {
            api_key = document.getElementById(`${provider}-api_key`)?.value || null;
            if (api_key == null)
                api_key = document.querySelector(`.${provider}-api_key`)?.value || null;
        }
        await api("conversation", {
            id: window.token,
            conversation_id: window.conversation_id,
            model: get_selected_model(),
            web_search: document.getElementById("switch").checked,
            provider: provider,
            messages: messages,
            auto_continue: auto_continue,
            api_key: api_key
        }, file);
        if (!error) {
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
            content_inner.innerHTML += `<p><strong>An error occured:</strong> ${e}</p>`;
        }
    }
    if (!error && text) {
        await add_message(window.conversation_id, "assistant", text, provider_result);
        await load_conversation(window.conversation_id);
    } else {
        let cursorDiv = document.getElementById("cursor");
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);
    }
    window.scrollTo(0, 0);
    message_box.scrollTop = message_box.scrollHeight;
    await remove_cancel_button();
    await register_message_buttons();
    prompt_lock = false;
    await load_conversations();
    regenerate.classList.remove("regenerate-hidden");
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

async function set_conversation_title(conversation_id, title) {
    conversation = await get_conversation(conversation_id)
    conversation.new_title = title;
    appStorage.setItem(
        `conversation:${conversation.id}`,
        JSON.stringify(conversation)
    );
}

const show_option = async (conversation_id) => {
    const conv = document.getElementById(`conv-${conversation_id}`);
    const choi = document.getElementById(`cho-${conversation_id}`);

    conv.style.display = "none";
    choi.style.display  = "block";

    const el = document.getElementById(`convo-${conversation_id}`);
    const trash_el = el.querySelector(".fa-trash");
    const title_el = el.querySelector("span.convo-title");
    if (title_el) {
        const left_el = el.querySelector(".left");
        const input_el = document.createElement("input");
        input_el.value = title_el.innerText;
        input_el.classList.add("convo-title");
        input_el.onfocus = () => trash_el.style.display = "none";
        input_el.onchange = () => set_conversation_title(conversation_id, input_el.value);
        left_el.removeChild(title_el);
        left_el.appendChild(input_el);
    }
};

const hide_option = async (conversation_id) => {
    const conv = document.getElementById(`conv-${conversation_id}`);
    const choi  = document.getElementById(`cho-${conversation_id}`);

    conv.style.display = "block";
    choi.style.display  = "none";

    const el = document.getElementById(`convo-${conversation_id}`);
    el.querySelector(".fa-trash").style.display = "";
    const input_el = el.querySelector("input.convo-title");
    if (input_el) {
        const left_el = el.querySelector(".left");
        const span_el = document.createElement("span");
        span_el.innerText = input_el.value;
        span_el.classList.add("convo-title");
        span_el.onclick = () => set_conversation(conversation_id);
        left_el.removeChild(input_el);
        left_el.appendChild(span_el);
    }
};

const delete_conversation = async (conversation_id) => {
    appStorage.removeItem(`conversation:${conversation_id}`);

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
    if (systemPrompt) {
        systemPrompt.value = "";
    }
    load_conversations();
    hide_sidebar();
    say_hello();
};

const load_conversation = async (conversation_id, scroll=true) => {
    let conversation = await get_conversation(conversation_id);
    let messages = conversation?.items || [];

    if (systemPrompt) {
        systemPrompt.value = conversation.system || "";
    }

    let elements = "";
    let last_model = null;
    for (i in messages) {
        let item = messages[i];
        last_model = item.provider?.model;
        let next_i = parseInt(i) + 1;
        let next_provider = item.provider ? item.provider : (messages.length > next_i ? messages[next_i].provider : null);
        let provider_label = item.provider?.label ? item.provider.label : item.provider?.name;
        let provider_link = item.provider?.name ? `<a href="${item.provider.url}" target="_blank">${provider_label}</a>` : "";
        let provider = provider_link ? `
            <div class="provider">
                ${provider_link}
                ${item.provider.model ? ' with ' + item.provider.model : ''}
            </div>
        ` : "";
        elements += `
            <div class="message${item.regenerate ? " regenerate": ""}" data-index="${i}">
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
                    <div class="count">
                        ${count_words_and_tokens(item.content, next_provider?.model)}
                        <i class="fa-solid fa-volume-high"></i>
                        <i class="fa-regular fa-clipboard"></i>
                        <a><i class="fa-brands fa-whatsapp"></i></a>
                        <i class="fa-solid fa-print"></i>
                    </div>
                </div>
            </div>
        `;
    }

    if (window.GPTTokenizer_cl100k_base) {
        const filtered = prepare_messages(messages, null);
        if (filtered.length > 0) {
            last_model = last_model?.startsWith("gpt-4") ? "gpt-4" : "gpt-3.5-turbo"
            let count_total = GPTTokenizer_cl100k_base?.encodeChat(filtered, last_model).length
            if (count_total > 0) {
                elements += `<div class="count_total">(${count_total} tokens used)</div>`;
            }
        }
    }

    message_box.innerHTML = elements;
    register_message_buttons();
    highlight(message_box);
    regenerate.classList.remove("regenerate-hidden");

    if (scroll) {
        message_box.scrollTo({ top: message_box.scrollHeight, behavior: "smooth" });

        setTimeout(() => {
            message_box.scrollTop = message_box.scrollHeight;
        }, 500);
    }
};

async function get_conversation(conversation_id) {
    let conversation = await JSON.parse(
        appStorage.getItem(`conversation:${conversation_id}`)
    );
    return conversation;
}

async function save_conversation(conversation_id, conversation) {
    conversation.updated = Date.now();
    appStorage.setItem(
        `conversation:${conversation_id}`,
        JSON.stringify(conversation)
    );
}

async function get_messages(conversation_id) {
    const conversation = await get_conversation(conversation_id);
    return conversation?.items || [];
}

async function add_conversation(conversation_id, content) {
    if (appStorage.getItem(`conversation:${conversation_id}`) == null) {
        await save_conversation(conversation_id, {
            id: conversation_id,
            title: "",
            added: Date.now(),
            system: systemPrompt?.value,
            items: [],
        });
    }
    history.pushState({}, null, `/chat/${conversation_id}`);
}

async function save_system_message() {
    if (!window.conversation_id) {
        return;
    }
    const conversation = await get_conversation(window.conversation_id);
    if (conversation) {
        conversation.system = systemPrompt?.value;
        await save_conversation(window.conversation_id, conversation);
    }
}
const hide_message = async (conversation_id, message_index =- 1) => {
    const conversation = await get_conversation(conversation_id)
    message_index = message_index == -1 ? conversation.items.length - 1 : message_index
    const last_message = message_index in conversation.items ? conversation.items[message_index] : null;
    if (last_message !== null) {
        if (last_message["role"] == "assistant") {
            last_message["regenerate"] = true;
        }
        conversation.items[message_index] = last_message;
    }
    await save_conversation(conversation_id, conversation);
};

const remove_message = async (conversation_id, index) => {
    const conversation = await get_conversation(conversation_id);
    let new_items = [];
    for (i in conversation.items) {
        if (i == index - 1) {
            if (!conversation.items[index]?.regenerate) {
                delete conversation.items[i]["regenerate"];
            }
        }
        if (i != index) {
            new_items.push(conversation.items[i])
        }
    }
    conversation.items = new_items;
    await save_conversation(conversation_id, conversation);
};

const get_message = async (conversation_id, index) => {
    const messages = await get_messages(conversation_id);
    if (index in messages)
        return messages[index]["content"];
};

const add_message = async (conversation_id, role, content, provider) => {
    const conversation = await get_conversation(conversation_id);
    conversation.items.push({
        role: role,
        content: content,
        provider: provider
    });
    await save_conversation(conversation_id, conversation);
    return conversation.items.length - 1;
};

const load_conversations = async () => {
    let conversations = [];
    for (let i = 0; i < appStorage.length; i++) {
        if (appStorage.key(i).startsWith("conversation:")) {
            let conversation = appStorage.getItem(appStorage.key(i));
            conversations.push(JSON.parse(conversation));
        }
    }
    conversations.sort((a, b) => (b.updated||0)-(a.updated||0));

    await clear_conversations();

    let html = "";
    conversations.forEach((conversation) => {
        if (conversation?.items.length > 0 && !conversation.new_title) {
            let new_value = (conversation.items[0]["content"]).trim();
            let new_lenght = new_value.indexOf("\n");
            new_lenght = new_lenght > 200 || new_lenght < 0 ? 200 : new_lenght;
            conversation.new_title = new_value.substring(0, new_lenght);
            appStorage.setItem(
                `conversation:${conversation.id}`,
                JSON.stringify(conversation)
            );
        }
        let updated = "";
        if (conversation.updated) {
            const date = new Date(conversation.updated);
            updated = date.toLocaleString('en-GB', {dateStyle: 'short', timeStyle: 'short', monthStyle: 'short'});
            updated = updated.replace("/" + date.getFullYear(), "")
        }
        html += `
            <div class="convo" id="convo-${conversation.id}">
                <div class="left">
                    <i class="fa-regular fa-comments"></i>
                    <span class="datetime" onclick="set_conversation('${conversation.id}')">${updated}</span>
                    <span class="convo-title" onclick="set_conversation('${conversation.id}')">${conversation.new_title}</span>
                </div>
                <i onclick="show_option('${conversation.id}')" class="fa-solid fa-ellipsis-vertical" id="conv-${conversation.id}"></i>
                <div id="cho-${conversation.id}" class="choise" style="display:none;">
                    <i onclick="delete_conversation('${conversation.id}')" class="fa-regular fa-trash"></i>
                    <i onclick="hide_option('${conversation.id}')" class="fa-regular fa-x"></i>
                </div>
            </div>
        `;
    });
    box_conversations.innerHTML += html;
};

document.getElementById("cancelButton").addEventListener("click", async () => {
    window.controller.abort();
    if (!window.abort) {
        window.abort = true;
        content_inner.innerHTML += " [aborted]";
        if (window.text) window.text += " [aborted]";
    }
    console.log(`aborted ${window.conversation_id}`);
});

document.getElementById("regenerateButton").addEventListener("click", async () => {
    prompt_lock = true;
    await hide_message(window.conversation_id);
    window.token = message_id();
    await ask_gpt();
});

const hide_input = document.querySelector(".toolbar .hide-input");
hide_input.addEventListener("click", async (e) => {
    const icon = hide_input.querySelector("i");
    const func = icon.classList.contains("fa-angles-down") ? "add" : "remove";
    const remv = icon.classList.contains("fa-angles-down") ? "remove" : "add";
    icon.classList[func]("fa-angles-up");
    icon.classList[remv]("fa-angles-down");
    document.querySelector(".conversation .user-input").classList[func]("hidden");
    document.querySelector(".conversation .buttons").classList[func]("hidden");
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
    settings.classList.add("hidden");
    chat.classList.remove("hidden");
    if (window.location.pathname == "/menu/" || window.location.pathname == "/settings/") {
        history.back();
    }
}

window.addEventListener('popstate', hide_sidebar, false);

sidebar_button.addEventListener("click", (event) => {
    settings.classList.add("hidden");
    if (sidebar.classList.contains("shown")) {
        hide_sidebar();
    } else {
        sidebar.classList.add("shown");
        sidebar_button.classList.add("rotated");
        history.pushState({}, null, "/menu/");
    }
    window.scrollTo(0, 0);
});

function open_settings() {
    if (settings.classList.contains("hidden")) {
        chat.classList.add("hidden");
        sidebar.classList.remove("shown");
        settings.classList.remove("hidden");
        history.pushState({}, null, "/settings/");
    } else {
        settings.classList.add("hidden");
        chat.classList.remove("hidden");
    }
}

function open_album() {
    if (album.classList.contains("hidden")) {
        sidebar.classList.remove("shown");
        settings.classList.add("hidden");
        album.classList.remove("hidden");
        history.pushState({}, null, "/images/");
    } else {
        album.classList.add("hidden");
    }
}

const register_settings_storage = async () => {
    optionElements.forEach((element) => {
        if (element.type == "textarea") {
            element.addEventListener('input', async (event) => {
                appStorage.setItem(element.id, element.value);
            });
        } else {
            element.addEventListener('change', async (event) => {
                switch (element.type) {
                    case "checkbox":
                        appStorage.setItem(element.id, element.checked);
                        break;
                    case "select-one":
                        appStorage.setItem(element.id, element.selectedIndex);
                        break;
                    case "text":
                    case "number":
                        appStorage.setItem(element.id, element.value);
                        break;
                    default:
                        console.warn("Unresolved element type");
                }
            });
        }
    });
}

const load_settings_storage = async () => {
    optionElements.forEach((element) => {
        if (!(value = appStorage.getItem(element.id))) {
            return;
        }
        if (value) {
            switch (element.type) {
                case "checkbox":
                    element.checked = value === "true";
                    break;
                case "select-one":
                    element.selectedIndex = parseInt(value);
                    break;
                case "text":
                case "number":
                case "textarea":
                    element.value = value;
                    break;
                default:
                    console.warn("Unresolved element type");
            }
        }
    });
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
    appStorage.setItem("theme", theme);
};

// set theme when visitor returns
const setTheme = function () {
    const activeTheme = appStorage.getItem("theme");
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

function count_tokens(model, text) {
    if (model) {
        if (window.llamaTokenizer)
        if (model.startsWith("llama") || model.startsWith("codellama")) {
            return llamaTokenizer.encode(text).length;
        }
        if (window.mistralTokenizer)
        if (model.startsWith("mistral") || model.startsWith("mixtral")) {
            return mistralTokenizer.encode(text).length;
        }
    }
    if (window.GPTTokenizer_cl100k_base) {
        return GPTTokenizer_cl100k_base.encode(text).length;
    }
}

function count_words(text) {
    return text.trim().match(/[\w\u4E00-\u9FA5]+/gu)?.length || 0;
}

function count_chars(text) {
    return text.match(/[^\s\p{P}]/gu)?.length || 0;
}

function count_words_and_tokens(text, model) {
    text = filter_message(text);
    return `(${count_words(text)} words, ${count_chars(text)} chars, ${count_tokens(model, text)} tokens)`;
}

let countFocus = messageInput;
let timeoutId;
const count_input = async () => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
        if (countFocus.value) {
            inputCount.innerText = count_words_and_tokens(countFocus.value, get_selected_model());
        } else {
            inputCount.innerText = "";
        }
    }, 100);
};
messageInput.addEventListener("keyup", count_input);
systemPrompt.addEventListener("keyup", count_input);
systemPrompt.addEventListener("focus", function() {
    countFocus = systemPrompt;
    count_input();
});
systemPrompt.addEventListener("input", function() {
    countFocus = messageInput;
    count_input();
});

window.addEventListener('load', async function() {
    await on_load();
    if (window.conversation_id == "{{chat_id}}") {
        window.conversation_id = uuid();
    } else {
        await on_api();
    }
});

window.addEventListener('pywebviewready', async function() {
    await on_api();
});

async function on_load() {
    setTheme();
    count_input();

    if (/\/chat\/.+/.test(window.location.href)) {
        load_conversation(window.conversation_id);
    } else {
        say_hello()
    }
    load_conversations();
}

async function on_api() {
    messageInput.addEventListener("keydown", async (evt) => {
        if (prompt_lock) return;

        // If not mobile
        if (!window.matchMedia("(pointer:coarse)").matches)
        if (evt.keyCode === 13 && !evt.shiftKey) {
            evt.preventDefault();
            console.log("pressed enter");
            await handle_ask();
        } else {
            messageInput.style.removeProperty("height");
            messageInput.style.height = messageInput.scrollHeight  + "px";
        }
    });
    sendButton.addEventListener(`click`, async () => {
        console.log("clicked send");
        if (prompt_lock) return;
        await handle_ask();
    });
    messageInput.focus();

    register_settings_storage();

    models = await api("models");
    models.forEach((model) => {
        let option = document.createElement("option");
        option.value = option.text = model;
        modelSelect.appendChild(option);
    });

    providers = await api("providers")
    Object.entries(providers).forEach(([provider, label]) => {
        let option = document.createElement("option");
        option.value = provider;
        option.text = label;
        providerSelect.appendChild(option);
    })

    await load_provider_models(appStorage.getItem("provider"));
    await load_settings_storage()

    const hide_systemPrompt = document.getElementById("hide-systemPrompt")
    const slide_systemPrompt_icon = document.querySelector(".slide-systemPrompt i");
    if (hide_systemPrompt.checked) {
        systemPrompt.classList.add("hidden");
        slide_systemPrompt_icon.classList.remove("fa-angles-up");
        slide_systemPrompt_icon.classList.add("fa-angles-down");
    }
    hide_systemPrompt.addEventListener('change', async (event) => {
        if (event.target.checked) {
            systemPrompt.classList.add("hidden");
        } else {
            systemPrompt.classList.remove("hidden");
        }
    });
    document.querySelector(".slide-systemPrompt")?.addEventListener("click", () => {
        hide_systemPrompt.click();
        let checked = hide_systemPrompt.checked;
        systemPrompt.classList[checked ? "add": "remove"]("hidden");
        slide_systemPrompt_icon.classList[checked ? "remove": "add"]("fa-angles-up");
        slide_systemPrompt_icon.classList[checked ? "add": "remove"]("fa-angles-down");
    });
    const messageInputHeight = document.getElementById("message-input-height");
    if (messageInputHeight) {
        if (messageInputHeight.value) {
            messageInput.style.maxHeight = `${messageInputHeight.value}px`;
        }
        messageInputHeight.addEventListener('change', async () => {
            messageInput.style.maxHeight = `${messageInputHeight.value}px`;
        });
    }
    const darkMode = document.getElementById("darkMode");
    if (darkMode) {
        if (!darkMode.checked) {
            document.body.classList.add("white");
        }
        darkMode.addEventListener('change', async (event) => {
            if (event.target.checked) {
                document.body.classList.remove("white");
            } else {
                document.body.classList.add("white");
            }
        });
    }
}

async function load_version() {
    const versions = await api("version");
    document.title = 'g4f - ' + versions["version"];
    let text = "version ~ "
    if (versions["version"] != versions["latest_version"]) {
        let release_url = 'https://github.com/xtekky/gpt4free/releases/tag/' + versions["latest_version"];
        let title = `New version: ${versions["latest_version"]}`;
        text += `<a href="${release_url}" target="_blank" title="${title}">${versions["version"]}</a> `;
        text += `<i class="fa-solid fa-rotate"></i>`
    } else {
        text += versions["version"];
    }
    document.getElementById("version_text").innerHTML = text
}
setTimeout(load_version, 2000);

[imageInput, cameraInput].forEach((el) => {
    el.addEventListener('click', async () => {
        el.value = '';
        if (imageInput.dataset.src) {
            URL.revokeObjectURL(imageInput.dataset.src);
            delete imageInput.dataset.src
        }
    });
});

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
        reader.addEventListener('load', async (event) => {
            fileInput.dataset.text = event.target.result;
            if (type == "json") {
                const data = JSON.parse(fileInput.dataset.text);
                if ("g4f" in data.options) {
                    Object.keys(data).forEach(key => {
                        if (key != "options" && !localStorage.getItem(key)) {
                            appStorage.setItem(key, JSON.stringify(data[key]));
                        } 
                    });
                    delete fileInput.dataset.text;
                    await load_conversations();
                    fileInput.value = "";
                }
            }
        });
        reader.readAsText(fileInput.files[0]);
    } else {
        delete fileInput.dataset.text;
    }
});

systemPrompt?.addEventListener("input", async () => {
    await save_system_message();
});

function get_selected_model() {
    if (modelProvider.selectedIndex >= 0) {
        return modelProvider.options[modelProvider.selectedIndex].value;
    } else if (modelSelect.selectedIndex >= 0) {
        return modelSelect.options[modelSelect.selectedIndex].value;
    }
}

async function api(ressource, args=null, file=null) {
    if (window?.pywebview) {
        if (args !== null) {
            if (ressource == "models") {
                ressource = "provider_models";
            }
            return pywebview.api[`get_${ressource}`](args);
        }
        return pywebview.api[`get_${ressource}`]();
    }
    if (ressource == "models" && args) {
        ressource = `${ressource}/${args}`;
    }
    const url = `/backend-api/v2/${ressource}`;
    if (ressource == "conversation") {
        let body = JSON.stringify(args);
        const headers = {
            accept: 'text/event-stream'
        }
        if (file !== null) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('json', body);
            body = formData;
        } else {
            headers['content-type'] = 'application/json';
        }
        response = await fetch(url, {
            method: 'POST',
            signal: window.controller.signal,
            headers: headers,
            body: body
        });
        return read_response(response);
    }
    response = await fetch(url);
    return await response.json();
}

async function read_response(response) {
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = ""
    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }
        for (const line of value.split("\n")) {
            if (!line) {
                continue;
            }
            try {
                add_message_chunk(JSON.parse(buffer + line))
                buffer = "";
            } catch {
                buffer += line
            }
        }
    }
}

async function load_provider_models(providerIndex=null) {
    if (!providerIndex) {
        providerIndex = providerSelect.selectedIndex;
    }
    modelProvider.innerHTML = '';
    const provider = providerSelect.options[providerIndex].value;
    if (!provider) {
        modelProvider.classList.add("hidden");
        modelSelect.classList.remove("hidden");
        return;
    }
    const models = await api('models', provider);
    if (models.length > 0) {
        modelSelect.classList.add("hidden");
        modelProvider.classList.remove("hidden");
        models.forEach((model) => {
            let option = document.createElement('option');
            option.value = option.text = model.model;
            option.selected = model.default;
            modelProvider.appendChild(option);
        });
    } else {
        modelProvider.classList.add("hidden");
        modelSelect.classList.remove("hidden");
    }
};
providerSelect.addEventListener("change", () => load_provider_models());

function save_storage() {
    let filename = `chat ${new Date().toLocaleString()}.json`.replaceAll(":", "-");
    let data = {"options": {"g4f": ""}};
    for (let i = 0; i < appStorage.length; i++){
        let key = appStorage.key(i);
        let item = appStorage.getItem(key);
        if (key.startsWith("conversation:")) {
            data[key] = JSON.parse(item);
        } else if (!key.includes("api_key")) {
            data["options"][key] = item;
        }
    }
    data = JSON.stringify(data, null, 4);
    const blob = new Blob([data], {type: 'application/json'});
    if(window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, filename);
    } else{
        const elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(blob);
        elem.download = filename;        
        document.body.appendChild(elem);
        elem.click();        
        document.body.removeChild(elem);
    }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const mircoIcon = microLabel.querySelector("i");
    mircoIcon.classList.add("fa-microphone");
    mircoIcon.classList.remove("fa-microphone-slash");

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    let startValue;
    let lastDebounceTranscript;
    recognition.onstart = function() {
        microLabel.classList.add("recognition");
        startValue = messageInput.value;
        lastDebounceTranscript = "";
    };
    recognition.onend = function() {
        messageInput.focus();
    };
    recognition.onresult = function(event) {
        if (!event.results) {
            return;
        }
        let result = event.results[event.resultIndex];
        let isFinal = result.isFinal && (result[0].confidence > 0);
        let transcript = result[0].transcript;
        if (isFinal) {
            if(transcript == lastDebounceTranscript) {
                return;
            }
            lastDebounceTranscript = transcript;
        }
        if (transcript) {
            messageInput.value = `${startValue ? startValue+"\n" : ""}${transcript.trim()}`;
            if (isFinal) {
                startValue = messageInput.value;
            }
            messageInput.style.height = messageInput.scrollHeight  + "px";
            messageInput.scrollTop = messageInput.scrollHeight;
        }
    };

    microLabel.addEventListener("click", () => {
        if (microLabel.classList.contains("recognition")) {
            recognition.stop();
            microLabel.classList.remove("recognition");
        } else {
            const lang = document.getElementById("recognition-language")?.value;
            recognition.lang = lang || navigator.language;
            recognition.start();
        }
    });
}