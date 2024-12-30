const colorThemes       = document.querySelectorAll('[name="theme"]');
const message_box       = document.getElementById(`messages`);
const messageInput      = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const stop_generating   = document.querySelector(`.stop_generating`);
const regenerate_button = document.querySelector(`.regenerate`);
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
const log_storage       = document.querySelector(".log");
const switchInput       = document.getElementById("switch");
const searchButton      = document.getElementById("search");

const optionElementsSelector = ".settings input, .settings textarea, #model, #model2, #provider";

let provider_storage = {};
let message_storage = {};
let controller_storage = {};
let content_storage = {};
let error_storage = {};
let synthesize_storage = {};
let title_storage = {};
let parameters_storage = {};
let finish_storage = {};

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

appStorage.getItem("darkMode") == "false" ? document.body.classList.add("white") : null;

let markdown_render = () => null;
if (window.markdownit) {
    const markdown = window.markdownit();
    markdown_render = (content) => {
        return markdown.render(content
            .replaceAll(/<!-- generated images start -->|<!-- generated images end -->/gm, "")
            .replaceAll(/<img data-prompt="[^>]+">/gm, "")
        )
            .replaceAll("<a href=", '<a target="_blank" href=')
            .replaceAll('<code>', '<code class="language-plaintext">')
    }
}

function filter_message(text) {
    return text.replaceAll(
        /<!-- generated images start -->[\s\S]+<!-- generated images end -->/gm, ""
    )
}

function fallback_clipboard (text) {
    var textBox = document.createElement("textarea");
    textBox.value = text;
    textBox.style.top = "0";
    textBox.style.left = "0";
    textBox.style.position = "fixed";
    document.body.appendChild(textBox);
    textBox.focus();
    textBox.select();
    try {
        var success = document.execCommand('copy');
        var msg = success ? 'succeeded' : 'failed';
        console.log('Clipboard Fallback: Copying text command ' + msg);
    } catch (e) {
        console.error('Clipboard Fallback: Unable to copy', e);
    }
    document.body.removeChild(textBox);
}

const iframe_container = Object.assign(document.createElement("div"), {
    className: "hljs-iframe-container hidden",
});
const iframe = Object.assign(document.createElement("iframe"), {
    className: "hljs-iframe",
});
iframe_container.appendChild(iframe);
const iframe_close = Object.assign(document.createElement("button"), {
    className: "hljs-iframe-close",
    innerHTML: '<i class="fa-regular fa-x"></i>',
});
iframe_close.onclick = () => iframe_container.classList.add("hidden");
iframe_container.appendChild(iframe_close);
chat.appendChild(iframe_container);

class HtmlRenderPlugin {
    constructor(options = {}) {
        self.hook = options.hook;
        self.callback = options.callback
    }
    "after:highlightElement"({
        el,
        text
    }) {
        if (!el.classList.contains("language-html")) {
            return;
        }
        let button = Object.assign(document.createElement("button"), {
            innerHTML: '<i class="fa-regular fa-folder-open"></i>',
            className: "hljs-iframe-button",
        });
        el.parentElement.appendChild(button);
        button.onclick = async () => {
            let newText = text;
            if (hook && typeof hook === "function") {
                newText = hook(text, el) || text
            }
            iframe.src = `data:text/html;charset=utf-8,${encodeURIComponent(newText)}`;
            iframe_container.classList.remove("hidden");
            if (typeof callback === "function") return callback(newText, el);
        }
    }
}

let typesetPromise = Promise.resolve();
const highlight = (container) => {
    if (window.hljs) {
        hljs.addPlugin(new HtmlRenderPlugin())
        if (window.CopyButtonPlugin) {
            hljs.addPlugin(new CopyButtonPlugin());
        }
        container.querySelectorAll('code:not(.hljs').forEach((el) => {
            if (el.className != "hljs") {
                hljs.highlightElement(el);
            }
        });
    }
    if (window.MathJax && window.MathJax.typesetPromise) {
        typesetPromise = typesetPromise.then(
            () => MathJax.typesetPromise([container])
        ).catch(
            (err) => console.log('Typeset failed: ' + err.message)
        );
    }
}

const get_message_el = (el) => {
    let message_el = el;
    while(!("index" in message_el.dataset) && message_el.parentElement) {
        message_el = message_el.parentElement;
    }
    return message_el;
}

const register_message_buttons = async () => {
    document.querySelectorAll(".message .content .provider").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            const provider_forms = document.querySelector(".provider_forms");
            const provider_form = provider_forms.querySelector(`#${el.dataset.provider}-form`);
            const provider_link = el.querySelector("a");
            provider_link?.addEventListener("click", async (event) => {
                event.preventDefault();
                if (provider_form) {
                    provider_form.classList.remove("hidden");
                    provider_forms.classList.remove("hidden");
                    chat.classList.add("hidden");
                }
                return false;
            });
            document.getElementById("close_provider_forms").addEventListener("click", async () => {
                provider_form.classList.add("hidden");
                provider_forms.classList.add("hidden");
                chat.classList.remove("hidden");
            });
        }
    });

    document.querySelectorAll(".message .fa-xmark").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = get_message_el(el);
                await remove_message(window.conversation_id, message_el.dataset.index);
                await safe_load_conversation(window.conversation_id, false);
            });
        }
    });

    document.querySelectorAll(".message .fa-clipboard").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                let message_el = get_message_el(el);
                const copyText = await get_message(window.conversation_id, message_el.dataset.index);
                try {        
                    if (!navigator.clipboard) {
                        throw new Error("navigator.clipboard: Clipboard API unavailable.");
                    }
                    await navigator.clipboard.writeText(copyText);
                } catch (e) {
                    console.error(e);
                    console.error("Clipboard API writeText() failed! Fallback to document.exec(\"copy\")...");
                    fallback_clipboard(copyText);
                }
                el.classList.add("clicked");
                setTimeout(() => el.classList.remove("clicked"), 1000);
            })
        }
    });

    document.querySelectorAll(".message .fa-volume-high").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = get_message_el(el);
                let audio;
                if (message_el.dataset.synthesize_url) {
                    el.classList.add("active");
                    setTimeout(()=>el.classList.remove("active"), 2000);
                    const media_player = document.querySelector(".media_player");
                    if (!media_player.classList.contains("show")) {
                        media_player.classList.add("show");
                        audio = new Audio(message_el.dataset.synthesize_url);
                        audio.controls = true;   
                        media_player.appendChild(audio);
                    } else {
                        audio = media_player.querySelector("audio");
                        audio.src = message_el.dataset.synthesize_url;
                    }
                    audio.play();
                    return;
                }
            });
        }
    });

    document.querySelectorAll(".message .regenerate_button").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = get_message_el(el);
                el.classList.add("clicked");
                setTimeout(() => el.classList.remove("clicked"), 1000);
                await ask_gpt(get_message_id(), message_el.dataset.index);
            });
        }
    });

    document.querySelectorAll(".message .continue_button").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                if (!el.disabled) {
                    el.disabled = true;
                    const message_el = get_message_el(el);
                    el.classList.add("clicked");
                    setTimeout(() => {el.classList.remove("clicked"); el.disabled = false}, 1000);
                    await ask_gpt(get_message_id(), message_el.dataset.index, false, null, null, "continue");
                }
            });
        }
    });

    document.querySelectorAll(".message .fa-whatsapp").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const text = get_message_el(el).innerText;
                window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank');
            });
        }
    });

    document.querySelectorAll(".message .fa-print").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = get_message_el(el);
                el.classList.add("clicked");
                message_box.scrollTop = 0;
                message_el.classList.add("print");
                setTimeout(() => {
                    el.classList.remove("clicked");
                    message_el.classList.remove("print");
                }, 1000);
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
    await scroll_to_bottom();

    let message = messageInput.value;
    if (message.length <= 0) {
        return;
    }
    messageInput.value = "";
    await count_input()
    await add_conversation(window.conversation_id);

    if ("text" in fileInput.dataset) {
        message += '\n```' + fileInput.dataset.type + '\n'; 
        message += fileInput.dataset.text;
        message += '\n```'
    }
    let message_index = await add_message(window.conversation_id, "user", message);
    let message_id = get_message_id();

    if (imageInput.dataset.objects) {
        imageInput.dataset.objects.split(" ").forEach((object)=>URL.revokeObjectURL(object))
        delete imageInput.dataset.objects;
    }
    const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput
    images = [];
    if (input.files.length > 0) {
        for (const file of input.files) {
            images.push(URL.createObjectURL(file));
        }
        imageInput.dataset.objects = images.join(" ");
    }
    message_box.innerHTML += `
        <div class="message" data-index="${message_index}">
            <div class="user">
                ${user_image}
                <i class="fa-solid fa-xmark"></i>
                <i class="fa-regular fa-phone-arrow-up-right"></i>
            </div>
            <div class="content" id="user_${message_id}"> 
                <div class="content_inner">
                ${markdown_render(message)}
                ${images.map((object)=>'<img src="' + object + '" alt="Image upload">').join("")}
                </div>
                <div class="count">
                    ${count_words_and_tokens(message, get_selected_model()?.value)}
                    <i class="fa-solid fa-volume-high"></i>
                    <i class="fa-regular fa-clipboard"></i>
                    <a><i class="fa-brands fa-whatsapp"></i></a>
                    <i class="fa-solid fa-print"></i>
                    <i class="fa-solid fa-rotate"></i>
                </div>
            </div>
        </div>
    `;
    highlight(message_box);

    const all_pinned = document.querySelectorAll(".buttons button.pinned")
    if (all_pinned.length > 0) {
        all_pinned.forEach((el, idx) => ask_gpt(
            idx == 0 ? message_id : get_message_id(),
            -1,
            idx != 0,
            el.dataset.provider,
            el.dataset.model
        ));
    } else {
        await ask_gpt(message_id);
    }
};

async function safe_remove_cancel_button() {
    for (let key in controller_storage) {
        if (!controller_storage[key].signal.aborted) {
            return;
        }
    }
    stop_generating.classList.add("stop_generating-hidden");
}

regenerate_button.addEventListener("click", async () => {
    regenerate_button.classList.add("regenerate-hidden");
    setTimeout(()=>regenerate_button.classList.remove("regenerate-hidden"), 3000);
    const all_pinned = document.querySelectorAll(".buttons button.pinned")
    if (all_pinned.length > 0) {
        all_pinned.forEach((el) => ask_gpt(get_message_id(), -1, true, el.dataset.provider, el.dataset.model, "variant"));
    } else {
        await ask_gpt(get_message_id(), -1, true, null, null, "variant");
    }
});

stop_generating.addEventListener("click", async () => {
    stop_generating.classList.add("stop_generating-hidden");
    regenerate_button.classList.remove("regenerate-hidden");
    let key;
    for (key in controller_storage) {
        if (!controller_storage[key].signal.aborted) {
            let message = message_storage[key];
            if (message) {
                content_storage[key].inner.innerHTML += " [aborted]";
                message_storage[key] += " [aborted]";
                console.log(`aborted ${window.conversation_id} #${key}`);
            }
            controller_storage[key].abort();
        }
    }
    await load_conversation(window.conversation_id, false);
});

document.querySelector(".media_player .fa-x").addEventListener("click", ()=>{
    const media_player = document.querySelector(".media_player");
    media_player.classList.remove("show");
    const audio = document.querySelector(".media_player audio");
    media_player.removeChild(audio);
});

const prepare_messages = (messages, message_index = -1, do_continue = false) => {
    messages = [ ...messages ]
    if (message_index != null) {
        // Removes messages after selected
        if (message_index >= 0) {
            messages = messages.filter((_, index) => message_index >= index);
        }
        // Removes none user messages at end
        if (!do_continue) {
            let last_message;
            while (last_message = messages.pop()) {
                if (last_message["role"] == "user") {
                    messages.push(last_message);
                    break;
                }
            }
        }
    }
    // Combine messages with same role
    let last_message;
    let new_messages = [];
    messages.forEach((message) => {
        message_copy = { ...message };
        if (last_message) {
            if (last_message["role"] == message["role"]) {
                message_copy["content"] = last_message["content"] + message_copy["content"];
                new_messages.pop();
            }
        }
        last_message = message_copy;
        new_messages.push(last_message);
    });
    messages = new_messages;

    // Insert system prompt as first message
    new_messages = [];
    if (systemPrompt?.value) {
        new_messages.push({
            "role": "system",
            "content": systemPrompt.value
        });
    }

    // Remove history, if it's selected
    if (document.getElementById('history')?.checked) {
        if (message_index == null) {
            messages = [messages.pop(), messages.pop()];
        } else {
            messages = [messages.pop()];
        }
    }

    messages.forEach((new_message) => {
        // Include only not regenerated messages
        if (new_message && !new_message.regenerate) {
            // Copy message first
            new_message = { ...new_message };
            // Remove generated images from history
            new_message.content = filter_message(new_message.content);
            // Remove internal fields
            delete new_message.provider;
            delete new_message.synthesize;
            delete new_message.finish;
            delete new_message.conversation;
            // Append message to new messages
            new_messages.push(new_message)
        }
    });

    return new_messages;
}

async function load_provider_parameters(provider) {
    let form_id = `${provider}-form`;
    if (!parameters_storage[provider]) {
        parameters_storage[provider] = JSON.parse(appStorage.getItem(form_id));
    }
    if (parameters_storage[provider]) {
        let provider_forms = document.querySelector(".provider_forms");
        let form_el = document.createElement("form");
        form_el.id = form_id;
        form_el.classList.add("hidden");
        appStorage.setItem(form_el.id, JSON.stringify(parameters_storage[provider]));
        let old_form = message_box.querySelector(`#${provider}-form`);
        if (old_form) {
            provider_forms.removeChild(old_form);
        }
        Object.entries(parameters_storage[provider]).forEach(([key, value]) => {
            let el_id = `${provider}-${key}`;
            let saved_value = appStorage.getItem(el_id);
            let input_el;
            let field_el;
            if (typeof value == "boolean") {
                field_el = document.createElement("div");
                field_el.classList.add("field");
                if (saved_value) {
                    field_el.classList.add("saved");
                    saved_value = saved_value == "true";
                } else {
                    saved_value = value;
                }
                field_el.innerHTML = `<span class="label">${key}:</span>
                <input type="checkbox" id="${el_id}" name="${provider}[${key}]">
                <label for="${el_id}" class="toogle" title=""></label>
                <i class="fa-solid fa-xmark"></i>`;
                form_el.appendChild(field_el);
                input_el = field_el.querySelector("input");
                input_el.checked = saved_value;
                input_el.dataset.checked = value ? "true" : "false";
                input_el.onchange = () => {
                    field_el.classList.add("saved");
                    appStorage.setItem(el_id, input_el.checked ? "true" : "false");
                }
            } else if (typeof value == "string" || typeof value == "object"|| typeof value == "number") {
                field_el = document.createElement("div");
                field_el.classList.add("field");
                field_el.classList.add("box");
                if (typeof value == "object") {
                    value = JSON.stringify(value, null, 4);
                }
                if (saved_value) {
                    field_el.classList.add("saved");
                } else {
                    saved_value = value;
                }
                let placeholder;
                if (key in ["api_key", "proof_token"]) {
                    placeholder = value.length >= 22 ? (value.substring(0, 10) + "*".repeat(8) + value.substring(value.length-10)) : value;
                } else {
                    placeholder = value;
                }
                field_el.innerHTML = `<label for="${el_id}" title="">${key}:</label>`;
                if (Number.isInteger(value)) {
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${provider}[${key}]" value="${escapeHtml(value)}" class="slider" min="0" max="4096" step="1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else if (typeof value == "number") {
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${provider}[${key}]" value="${escapeHtml(value)}" class="slider" min="0" max="2" step="0.1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else {
                    field_el.innerHTML += `<textarea id="${el_id}" name="${provider}[${key}]"></textarea>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                    input_el = field_el.querySelector("textarea");
                    input_el.dataset.text = value;
                    input_el.placeholder = placeholder;
                    if (!key in ["api_key", "proof_token"]) {
                        input_el.innerHTML = saved_value;
                    } else {
                        input_el.dataset.saved_value = saved_value;
                    }
                    input_el.oninput = () => {
                        field_el.classList.add("saved");
                        appStorage.setItem(el_id, input_el.value);
                        input_el.dataset.saved_value = input_el.value;
                    };
                    input_el.onfocus = () => {
                        if (input_el.dataset.saved_value) {
                            input_el.innerHTML = input_el.dataset.saved_value;
                        }
                        input_el.style.removeProperty("height");
                        input_el.style.height = (input_el.scrollHeight) + "px";
                    }
                    input_el.onblur = () => {
                        input_el.style.removeProperty("height");
                        if (key in ["api_key", "proof_token"]) {
                            input_el.value = "";
                        }
                    }
                }
                if (!input_el) {
                    input_el = field_el.querySelector("input");
                    input_el.dataset.value = value;
                    input_el.value = saved_value;
                    input_el.oninput = () => {
                        input_el.nextElementSibling.value = input_el.value;
                        field_el.classList.add("saved");
                        appStorage.setItem(input_el.id, input_el.value);
                    };
                }
            }
            form_el.appendChild(field_el);
            let xmark_el = field_el.querySelector(".fa-xmark");
            xmark_el.onclick = () => {
                if (input_el.dataset.checked) {
                    input_el.checked = input_el.dataset.checked == "true";
                } else if (input_el.dataset.value) {
                    input_el.value = input_el.dataset.value;
                    input_el.nextElementSibling.value = input_el.dataset.value;
                } else if (input_el.dataset.text) {
                    input_el.innerHTML = input_el.dataset.text;
                }
                appStorage.removeItem(el_id);
                field_el.classList.remove("saved");
            }
        });
        provider_forms.prepend(form_el);
    }
}

async function add_message_chunk(message, message_id, provider) {
    content_map = content_storage[message_id];
    if (message.type == "conversation") {
        const conversation = await get_conversation(window.conversation_id);
        if (!conversation.data) {
            conversation.data = {};
        }
        for (const [key, value] of Object.entries(message.conversation)) {
            conversation.data[key] = value;
        }
        await save_conversation(conversation_id, conversation);
    } else if (message.type == "provider") {
        provider_storage[message_id] = message.provider;
        let provider_el = content_map.content.querySelector('.provider');
        provider_el.innerHTML = `
            <a href="${message.provider.url}" target="_blank">
                ${message.provider.label ? message.provider.label : message.provider.name}
            </a>
            ${message.provider.model ? ' with ' + message.provider.model : ''}
        `;
    } else if (message.type == "message") {
        console.error(message.message)
    } else if (message.type == "error") {
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
        error_storage[message_id] = message.error
        console.error(message.error);
        content_map.inner.innerHTML += markdown_render(`**An error occured:** ${message.error}`);
        let p = document.createElement("p");
        p.innerText = message.error;
        log_storage.appendChild(p);
    } else if (message.type == "preview") {
        if (content_map.inner.clientHeight > 200)
            content_map.inner.style.height = content_map.inner.clientHeight + "px";
        if (img = content_map.inner.querySelector("img"))
            if (!img.complete)
                return;
        content_map.inner.innerHTML = markdown_render(message.preview);
    } else if (message.type == "content") {
        message_storage[message_id] += message.content;
        update_message(content_map, message_id);
        content_map.inner.style.height = "";
    } else if (message.type == "log") {
        let p = document.createElement("p");
        p.innerText = message.log;
        log_storage.appendChild(p);
    } else if (message.type == "synthesize") {
        synthesize_storage[message_id] = message.synthesize;
    } else if (message.type == "title") {
        title_storage[message_id] = message.title;
    } else if (message.type == "login") {
        update_message(content_map, message_id, message.login);
    } else if (message.type == "login") {
        update_message(content_map, message_id, message.login);
    } else if (message.type == "finish") {
        finish_storage[message_id] = message.finish;
    } else if (message.type == "parameters") {
        if (!parameters_storage[provider]) {
            parameters_storage[provider] = {};
        }
        Object.entries(message.parameters).forEach(([key, value]) => {
            parameters_storage[provider][key] = value;
        });
        await load_provider_parameters(provider);
    }
}

const ask_gpt = async (message_id, message_index = -1, regenerate = false, provider = null, model = null, action = null) => {
    if (!model && !provider) {
        model = get_selected_model()?.value || null;
        provider = providerSelect.options[providerSelect.selectedIndex].value;
    }
    let conversation = await get_conversation(window.conversation_id);
    messages = prepare_messages(conversation.items, message_index, action=="continue");
    message_storage[message_id] = "";
    stop_generating.classList.remove("stop_generating-hidden");

    if (message_index == -1 && !regenerate) {
        await scroll_to_bottom();
    }

    let count_total = message_box.querySelector('.count_total');
    count_total ? count_total.parentElement.removeChild(count_total) : null;

    const message_el = document.createElement("div");
    message_el.classList.add("message");
    if (message_index != -1 || regenerate) {
        message_el.classList.add("regenerate");
    }
    message_el.innerHTML += `
        <div class="assistant">
            ${gpt_image}
            <i class="fa-solid fa-xmark"></i>
            <i class="fa-regular fa-phone-arrow-down-left"></i>
        </div>
        <div class="content" id="gpt_${message_id}">
            <div class="provider" data-provider="${provider}"></div>
            <div class="content_inner"><span class="cursor"></span></div>
            <div class="count"></div>
        </div>
    `;
    if (message_index == -1) {
        message_box.appendChild(message_el);
    } else {
        parent_message = message_box.querySelector(`.message[data-index="${message_index}"]`);
        if (!parent_message) {
            return;
        }
        parent_message.after(message_el);
    }

    controller_storage[message_id] = new AbortController();

    let content_el = document.getElementById(`gpt_${message_id}`)
    let content_map = content_storage[message_id] = {
        container: message_el,
        content: content_el,
        inner: content_el.querySelector('.content_inner'),
        count: content_el.querySelector('.count'),
        update_timeouts: [],
    }
    if (message_index == -1 && !regenerate) {
        await scroll_to_bottom();
    }
    try {
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput;
        const files = input && input.files.length > 0 ? input.files : null;
        const download_images = document.getElementById("download_images")?.checked;
        const api_key = get_api_key_by_provider(provider);
        const ignored = Array.from(settings.querySelectorAll("input.provider:not(:checked)")).map((el)=>el.value);
        await api("conversation", {
            id: message_id,
            conversation_id: window.conversation_id,
            conversation: conversation.data && provider in conversation.data ? conversation.data[provider] : null,
            model: model,
            web_search: switchInput.checked,
            provider: provider,
            messages: messages,
            action: action,
            download_images: download_images,
            api_key: api_key,
            ignored: ignored,
        }, files, message_id);
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
        if (!error_storage[message_id]) {
            html = markdown_render(message_storage[message_id]);
            content_map.inner.innerHTML = html;
            highlight(content_map.inner);
            if (imageInput) imageInput.value = "";
            if (cameraInput) cameraInput.value = "";
            if (fileInput) fileInput.value = "";
        }
    } catch (e) {
        console.error(e);
        if (e.name != "AbortError") {
            error_storage[message_id] = true;
            content_map.inner.innerHTML += markdown_render(`**An error occured:** ${e}`);
        }
    }
    delete controller_storage[message_id];
    if (message_storage[message_id]) {
        const message_provider = message_id in provider_storage ? provider_storage[message_id] : null;
        await add_message(
            window.conversation_id,
            "assistant",
            message_storage[message_id] + (error_storage[message_id] ? " [error]" : ""),
            message_provider,
            message_index,
            synthesize_storage[message_id],
            regenerate,
            title_storage[message_id],
            finish_storage[message_id],
            action=="continue"
        );
        delete message_storage[message_id];
        if (!error_storage[message_id]) {
            await safe_load_conversation(window.conversation_id, message_index == -1);
        }
    }
    let cursorDiv = message_el.querySelector(".cursor");
    if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);
    if (message_index == -1) {
        await scroll_to_bottom();
    }
    await safe_remove_cancel_button();
    await register_message_buttons();
    await load_conversations();
    regenerate_button.classList.remove("regenerate-hidden");
};

async function scroll_to_bottom() {
    window.scrollTo(0, 0);
    message_box.scrollTop = message_box.scrollHeight;
}

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
        input_el.onclick = (e) => e.stopPropagation()
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
    try {
        history.pushState({}, null, `/chat/${conversation_id}`);
    } catch (e) {
        console.error(e);
    }
    window.conversation_id = conversation_id;

    await clear_conversation();
    await load_conversation(conversation_id);
    load_conversations();
    hide_sidebar();
};

const new_conversation = async () => {
    history.pushState({}, null, `/chat/`);
    window.conversation_id = uuid();
    document.title = window.title || document.title;

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

    if (!conversation) {
        return;
    }
    let title = conversation.title || conversation.new_title;
    title = title ? `${title} - g4f` : window.title;
    if (title) {
        document.title = title;
    }

    if (systemPrompt) {
        systemPrompt.value = conversation.system || "";
    }

    let elements = [];
    let last_model = null;
    let providers = [];
    let buffer = "";
    messages.forEach((item, i) => {
        if (item.continue) {
            elements.pop();
        } else {
            buffer = "";
        }
        buffer += item.content;
        last_model = item.provider?.model;
        providers.push(item.provider?.name);
        let next_i = parseInt(i) + 1;
        let next_provider = item.provider ? item.provider : (messages.length > next_i ? messages[next_i].provider : null);
        let provider_label = item.provider?.label ? item.provider.label : item.provider?.name;
        let provider_link = item.provider?.name ? `<a href="${item.provider.url}" target="_blank">${provider_label}</a>` : "";
        let provider = provider_link ? `
            <div class="provider" data-provider="${item.provider.name}">
                ${provider_link}
                ${item.provider.model ? ' with ' + item.provider.model : ''}
            </div>
        ` : "";
        let synthesize_params = {text: buffer}
        let synthesize_provider = "Gemini";
        if (item.synthesize) {
            synthesize_params = item.synthesize.data
            synthesize_provider = item.synthesize.provider;
        }
        synthesize_params = (new URLSearchParams(synthesize_params)).toString();
        let synthesize_url = `/backend-api/v2/synthesize/${synthesize_provider}?${synthesize_params}`;

        let add_buttons = [];
        // Always add regenerate button
        add_buttons.push(`<button class="regenerate_button">
            <span>Regenerate</span>
            <i class="fa-solid fa-rotate"></i>
        </button>`);
        // Add continue button if possible
        if (item.finish && item.finish.actions) {
            item.finish.actions.forEach((action) => {
                if (action == "continue") {
                    if (messages.length >= i - 1) {
                        add_buttons.push(`<button class="continue_button">
                            <span>Continue</span>
                            <i class="fa-solid fa-wand-magic-sparkles"></i>
                        </button>`);
                    }
                }
            });
        }

        elements.push(`
            <div class="message${item.regenerate ? " regenerate": ""}" data-index="${i}" data-synthesize_url="${synthesize_url}">
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
                    <div class="content_inner">${markdown_render(buffer)}</div>
                    <div class="count">
                        ${count_words_and_tokens(buffer, next_provider?.model)}
                        <span>
                        <i class="fa-solid fa-volume-high"></i>
                        <i class="fa-regular fa-clipboard"></i>
                        <a><i class="fa-brands fa-whatsapp"></i></a>
                        <i class="fa-solid fa-print"></i>
                        </span>
                        ${add_buttons.join("")}
                    </div>
                </div>
            </div>
        `);
    });

    if (window.GPTTokenizer_cl100k_base) {
        const filtered = prepare_messages(messages, null);
        if (filtered.length > 0) {
            last_model = last_model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
            let count_total = GPTTokenizer_cl100k_base?.encodeChat(filtered, last_model).length
            if (count_total > 0) {
                elements.push(`<div class="count_total">(${count_total} tokens used)</div>`);
            }
        }
    }

    message_box.innerHTML = elements.join("");
    [...new Set(providers)].forEach(async (provider) => {
        await load_provider_parameters(provider);
    });
    register_message_buttons();
    highlight(message_box);
    regenerate_button.classList.remove("regenerate-hidden");

    if (scroll) {
        message_box.scrollTo({ top: message_box.scrollHeight, behavior: "smooth" });

        setTimeout(() => {
            message_box.scrollTop = message_box.scrollHeight;
        }, 500);
    }
};

async function safe_load_conversation(conversation_id, scroll=true) {
    let is_running = false
    for (const key in controller_storage) {
        if (!controller_storage[key].signal.aborted) {
            is_running = true;
            break
        }
    }
    if (!is_running) {
        load_conversation(conversation_id, scroll);
    }
}

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

async function add_conversation(conversation_id) {
    if (appStorage.getItem(`conversation:${conversation_id}`) == null) {
        await save_conversation(conversation_id, {
            id: conversation_id,
            title: "",
            added: Date.now(),
            system: systemPrompt?.value,
            items: [],
        });
    }
    try {
        history.pushState({}, null, `/chat/${conversation_id}`);
    } catch (e) {
        console.error(e);
    }
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

const add_message = async (
    conversation_id, role, content,
    provider = null,
    message_index = -1,
    synthesize_data = null,
    regenerate = false,
    title = null,
    finish = null,
    do_continue = false
) => {
    const conversation = await get_conversation(conversation_id);
    if (!conversation) {
        return;
    }
    if (title) {
        conversation.title = title;
    } else if (!conversation.title) {
        let new_value = content.trim();
        let new_lenght = new_value.indexOf("\n");
        new_lenght = new_lenght > 200 || new_lenght < 0 ? 200 : new_lenght;
        conversation.title = new_value.substring(0, new_lenght);
    }
    const new_message = {
        role: role,
        content: content,
        provider: provider,
    };
    if (synthesize_data) {
        new_message.synthesize = synthesize_data;
    }
    if (regenerate) {
        new_message.regenerate = true;
    }
    if (finish) {
        new_message.finish = finish;
    }
    if (do_continue) {
        new_message.continue = true;
    }
    if (message_index == -1) {
         conversation.items.push(new_message);
    } else {
        const new_messages = [];
        conversation.items.forEach((item, index)=>{
            new_messages.push(item);
            if (index == message_index) {
                new_messages.push(new_message);
            }
        });
        conversation.items = new_messages;
    }
    await save_conversation(conversation_id, conversation);
    return conversation.items.length - 1;
};

const escapeHtml = (unsafe) => {
    return unsafe+"".replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

const toLocaleDateString = (date) => {
    date = new Date(date);
    return date.toLocaleString('en-GB', {dateStyle: 'short', timeStyle: 'short', monthStyle: 'short'}).replace("/" + date.getFullYear(), "");
}

const load_conversations = async () => {
    let conversations = [];
    for (let i = 0; i < appStorage.length; i++) {
        if (appStorage.key(i).startsWith("conversation:")) {
            let conversation = appStorage.getItem(appStorage.key(i));
            conversations.push(JSON.parse(conversation));
        }
    }
    conversations.sort((a, b) => (b.updated||0)-(a.updated||0));

    let html = [];
    conversations.forEach((conversation) => {
        html.push(`
            <div class="convo" id="convo-${conversation.id}">
                <div class="left" onclick="set_conversation('${conversation.id}')">
                    <i class="fa-regular fa-comments"></i>
                    <span class="datetime">${conversation.updated ? toLocaleDateString(conversation.updated) : ""}</span>
                    <span class="convo-title">${escapeHtml(conversation.new_title ? conversation.new_title : conversation.title)}</span>
                </div>
                <i onclick="show_option('${conversation.id}')" class="fa-solid fa-ellipsis-vertical" id="conv-${conversation.id}"></i>
                <div id="cho-${conversation.id}" class="choise" style="display:none;">
                    <i onclick="delete_conversation('${conversation.id}')" class="fa-regular fa-trash"></i>
                    <i onclick="hide_option('${conversation.id}')" class="fa-regular fa-x"></i>
                </div>
            </div>
        `);
    });
    await clear_conversations();
    box_conversations.innerHTML += html.join("");
};

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

function get_message_id() {
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
    log_storage.classList.add("hidden");
    if (window.location.pathname == "/menu/" || window.location.pathname == "/settings/") {
        history.back();
    }
}

window.addEventListener('popstate', hide_sidebar, false);

sidebar_button.addEventListener("click", async () => {
    settings.classList.add("hidden");
    if (sidebar.classList.contains("shown")) {
        await hide_sidebar();
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
    log_storage.classList.add("hidden");
}

const register_settings_storage = async () => {
    const optionElements = document.querySelectorAll(optionElementsSelector);
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
    const optionElements = document.querySelectorAll(optionElementsSelector);
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
    return 0;
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

function update_message(content_map, message_id, content = null) {
    content_map.update_timeouts.push(setTimeout(() => {
        if (!content) content = message_storage[message_id];
        html = markdown_render(content);
        let lastElement, lastIndex = null;
        for (element of ['</p>', '</code></pre>', '</p>\n</li>\n</ol>', '</li>\n</ol>', '</li>\n</ul>']) {
            const index = html.lastIndexOf(element)
            if (index - element.length > lastIndex) {
                lastElement = element;
                lastIndex = index;
            }
        }
        if (lastIndex) {
            html = html.substring(0, lastIndex) + '<span class="cursor"></span>' + lastElement;
        }
        if (error_storage[message_id]) {
            content_map.inner.innerHTML += markdown_render(`**An error occured:** ${error_storage[message_id]}`);
        }
        content_map.inner.innerHTML = html;
        content_map.count.innerText = count_words_and_tokens(message_storage[message_id], provider_storage[message_id]?.model);
        highlight(content_map.inner);
        if (!content_map.container.classList.contains("regenerate")) {
            if (message_box.scrollTop >= message_box.scrollHeight - message_box.clientHeight - 200) {
                window.scrollTo(0, 0);
                message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
            }
        }
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
    }, 100));
};

let countFocus = messageInput;
let timeoutId;
const count_input = async () => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
        if (countFocus.value) {
            inputCount.innerText = count_words_and_tokens(countFocus.value, get_selected_model()?.value);
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
    await safe_load_conversation(window.conversation_id, false);
});

window.addEventListener('DOMContentLoaded', async function() {
    await on_load();
    if (window.conversation_id == "{{chat_id}}") {
        window.conversation_id = uuid();
    }
    await on_api();
});

async function on_load() {
    count_input();
    if (/\/chat\/.+/.test(window.location.href)) {
        load_conversation(window.conversation_id);
    } else {
        say_hello()
    }
    load_conversations();
}

const load_provider_option = (input, provider_name) => {
    if (input.checked) {
        modelSelect.querySelectorAll(`option[data-disabled_providers*="${provider_name}"]`).forEach(
            (el) => {
                el.dataset.disabled_providers = el.dataset.disabled_providers ? el.dataset.disabled_providers.split(" ").filter((provider) => provider!=provider_name).join(" ") : "";
                el.dataset.providers = (el.dataset.providers ? el.dataset.providers + " " : "") + provider_name;
                modelSelect.querySelectorAll(`option[value="${el.value}"]`).forEach((o)=>o.removeAttribute("disabled", "disabled"))
            }
        );
        providerSelect.querySelectorAll(`option[value="${provider_name}"]`).forEach(
            (el) => el.removeAttribute("disabled")
        );
        providerSelect.querySelectorAll(`option[data-parent="${provider_name}"]`).forEach(
            (el) => el.removeAttribute("disabled")
        );
        settings.querySelector(`.field:has(#${provider_name}-api_key)`)?.classList.remove("hidden");
    } else {
        modelSelect.querySelectorAll(`option[data-providers*="${provider_name}"]`).forEach(
            (el) => {
                el.dataset.providers = el.dataset.providers ? el.dataset.providers.split(" ").filter((provider) => provider!=provider_name).join(" ") : "";
                el.dataset.disabled_providers = (el.dataset.disabled_providers ? el.dataset.disabled_providers + " " : "") + provider_name;
                if (!el.dataset.providers) modelSelect.querySelectorAll(`option[value="${el.value}"]`).forEach((o)=>o.setAttribute("disabled", "disabled"))
            }
        );
        providerSelect.querySelectorAll(`option[value="${provider_name}"]`).forEach(
            (el) => el.setAttribute("disabled", "disabled")
        );
        providerSelect.querySelectorAll(`option[data-parent="${provider_name}"]`).forEach(
            (el) => el.setAttribute("disabled", "disabled")
        );
        settings.querySelector(`.field:has(#${provider_name}-api_key)`)?.classList.add("hidden");
    }
};

async function on_api() {
    let prompt_lock = false;
    messageInput.addEventListener("keydown", async (evt) => {
        if (prompt_lock) return;
        // If not mobile and not shift enter
        if (!window.matchMedia("(pointer:coarse)").matches && evt.keyCode === 13 && !evt.shiftKey) {
            evt.preventDefault();
            console.log("pressed enter");
            prompt_lock = true;
            setTimeout(()=>prompt_lock=false, 3000);
            await handle_ask();
        } else {
            messageInput.style.removeProperty("height");
            messageInput.style.height = messageInput.scrollHeight  + "px";
        }
    });
    sendButton.addEventListener(`click`, async () => {
        console.log("clicked send");
        if (prompt_lock) return;
        prompt_lock = true;
        setTimeout(()=>prompt_lock=false, 3000);
        await handle_ask();
    });
    messageInput.focus();
    let provider_options = [];
    try {
        models = await api("models");
        models.forEach((model) => {
            let option = document.createElement("option");
            option.value = model.name;
            option.text = model.name + (model.image ? " (Image Generation)" : "");
            option.dataset.providers = model.providers.join(" ");
            modelSelect.appendChild(option);
        });
        providers = await api("providers")
        providers.sort((a, b) => a.label.localeCompare(b.label));
        providers.forEach((provider) => {
            let option = document.createElement("option");
            option.value = provider.name;
            option.dataset.label = provider.label;
            option.text = provider.label
                + (provider.vision ? " (Image Upload)" : "")
                + (provider.image ? " (Image Generation)" : "")
                + (provider.webdriver ? " (Webdriver)" : "")
                + (provider.auth ? " (Auth)" : "");
            if (provider.parent)
                option.dataset.parent = provider.parent;
            providerSelect.appendChild(option);

            if (!provider.parent) {
                option = document.createElement("div");
                option.classList.add("field");
                option.innerHTML = `
                    <span class="label">Enable ${provider.label}</span>
                    <input id="Provider${provider.name}" type="checkbox" name="Provider${provider.name}" value="${provider.name}" class="provider" checked="">
                    <label for="Provider${provider.name}" class="toogle" title="Remove provider from dropdown"></label>
                `;
                option.querySelector("input").addEventListener("change", (event) => load_provider_option(event.target, provider.name));
                settings.querySelector(".paper").appendChild(option);
                provider_options[provider.name] = option;
            }
        });
        await load_provider_models(appStorage.getItem("provider"));
    } catch (e) {
        console.error(e)
        // Redirect to show basic authenfication
        if (document.location.pathname == "/chat/") {
            document.location.href = `/chat/error`;
        }
    }
    register_settings_storage();
    await load_settings_storage()
    Object.entries(provider_options).forEach(
        ([provider_name, option]) => load_provider_option(option.querySelector("input"), provider_name)
    );

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
        const checked = hide_systemPrompt.checked;
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
        darkMode.addEventListener('change', async (event) => {
            if (event.target.checked) {
                document.body.classList.remove("white");
            } else {
                document.body.classList.add("white");
            }
        });
    }

    const method = switchInput.checked ? "add" : "remove";
    searchButton.classList[method]("active");
}

async function load_version() {
    let new_version = document.querySelector(".new_version");
    if (new_version) return;
    const versions = await api("version");
    window.title = 'g4f - ' + versions["version"];
    if (document.title == "g4f - gui") {
        document.title = window.title;
    }
    let text = "version ~ "
    if (versions["version"] != versions["latest_version"]) {
        let release_url = 'https://github.com/xtekky/gpt4free/releases/latest';
        let title = `New version: ${versions["latest_version"]}`;
        text += `<a href="${release_url}" target="_blank" title="${title}">${versions["version"]}</a> 🆕`;
        new_version = document.createElement("div");
        new_version.classList.add("new_version");
        const link = `<a href="${release_url}" target="_blank" title="${title}">v${versions["latest_version"]}</a>`;
        new_version.innerHTML = `g4f ${link}&nbsp;&nbsp;🆕`;
        new_version.addEventListener("click", ()=>new_version.parentElement.removeChild(new_version));
        document.body.appendChild(new_version);
    } else {
        text += versions["version"];
    }
    document.getElementById("version_text").innerHTML = text
    setTimeout(load_version, 1000 * 60 * 60); // 1 hour
}
setTimeout(load_version, 100);

[imageInput, cameraInput].forEach((el) => {
    el.addEventListener('click', async () => {
        el.value = '';
        if (imageInput.dataset.objects) {
            imageInput.dataset.objects.split(" ").forEach((object) => URL.revokeObjectURL(object));
            delete imageInput.dataset.objects
        }
    });
});

fileInput.addEventListener('click', async (event) => {
    fileInput.value = '';
    delete fileInput.dataset.text;
});

async function upload_cookies() {
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    response = await fetch("/backend-api/v2/upload_cookies", {
        method: 'POST',
        body: formData,
    });
    if (response.status == 200) {
        inputCount.innerText = `${file.name} was uploaded successfully`;
    }
    fileInput.value = "";
}

fileInput.addEventListener('change', async (event) => {
    if (fileInput.files.length) {
        type = fileInput.files[0].name.split('.').pop()
        if (type == "har") {
            return await upload_cookies();
        }
        fileInput.dataset.type = type
        const reader = new FileReader();
        reader.addEventListener('load', async (event) => {
            fileInput.dataset.text = event.target.result;
            if (type == "json") {
                const data = JSON.parse(fileInput.dataset.text);
                if ("g4f" in data.options) {
                    let count = 0;
                    Object.keys(data).forEach(key => {
                        if (key != "options" && !localStorage.getItem(key)) {
                            appStorage.setItem(key, JSON.stringify(data[key]));
                            count += 1;
                        }
                    });
                    delete fileInput.dataset.text;
                    await load_conversations();
                    fileInput.value = "";
                    inputCount.innerText = `${count} Conversations were imported successfully`;
                } else {
                    await upload_cookies();
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
        return modelProvider.options[modelProvider.selectedIndex];
    } else if (modelSelect.selectedIndex >= 0) {
        model = modelSelect.options[modelSelect.selectedIndex];
        if (model.value) {
            return model;
        }
    }
}

async function api(ressource, args=null, files=null, message_id=null) {
    let api_key;
    if (ressource == "models" && args) {
        api_key = get_api_key_by_provider(args);
        ressource = `${ressource}/${args}`;
    }
    const url = `/backend-api/v2/${ressource}`;
    const headers = {};
    if (api_key) {
        headers.x_api_key = api_key;
    }
    if (ressource == "conversation") {
        let body = JSON.stringify(args);
        headers.accept = 'text/event-stream';
        if (files !== null) {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files[]', file)
            }
            formData.append('json', body);
            body = formData;
        } else {
            headers['content-type'] = 'application/json';
        }
        response = await fetch(url, {
            method: 'POST',
            signal: controller_storage[message_id].signal,
            headers: headers,
            body: body,
        });
        return read_response(response, message_id, args.provider || null);
    }
    response = await fetch(url, {headers: headers});
    return await response.json();
}

async function read_response(response, message_id, provider) {
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
                add_message_chunk(JSON.parse(buffer + line), message_id, provider);
                buffer = "";
            } catch {
                buffer += line
            }
        }
    }
}

function get_api_key_by_provider(provider) {
    let api_key = null;
    if (provider) {
        api_key = document.getElementById(`${provider}-api_key`)?.value || null;
        if (api_key == null) {
            api_key = document.querySelector(`.${provider}-api_key`)?.value || null;
        }
    }
    return api_key;
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
            option.value = model.model;
            option.dataset.label = model.model;
            option.text = `${model.model}${model.image ? " (Image Generation)" : ""}${model.vision ? " (Image Upload)" : ""}`;
            option.selected = model.default;
            modelProvider.appendChild(option);
        });
    } else {
        modelProvider.classList.add("hidden");
        modelSelect.classList.remove("hidden");
    }
};
providerSelect.addEventListener("change", () => load_provider_models());

document.getElementById("pin").addEventListener("click", async () => {
    const pin_container = document.getElementById("pin_container");
    let selected_provider = providerSelect.options[providerSelect.selectedIndex];
    selected_provider = selected_provider.value ? selected_provider : null;
    const selected_model = get_selected_model();
    if (selected_provider || selected_model) {
        const pinned = document.createElement("button");
        pinned.classList.add("pinned");
        if (selected_provider) pinned.dataset.provider = selected_provider.value;
        if (selected_model) pinned.dataset.model = selected_model.value;
        pinned.innerHTML = `
            <span>
            ${selected_provider ? selected_provider.dataset.label || selected_provider.text : ""}
            ${selected_provider && selected_model ? "/" : ""}
            ${selected_model ? selected_model.dataset.label || selected_model.text : ""}
            </span>
            <i class="fa-regular fa-circle-xmark"></i>`;
        pinned.addEventListener("click", () => pin_container.removeChild(pinned));
        let all_pinned = pin_container.querySelectorAll(".pinned");
        while (all_pinned.length > 4) {
            pin_container.removeChild(all_pinned[0])
            all_pinned = pin_container.querySelectorAll(".pinned");
        }
        pin_container.appendChild(pinned);
    }
});

switchInput.addEventListener("change", () => {
    const method = switchInput.checked ? "add" : "remove";
    searchButton.classList[method]("active");
});
searchButton.addEventListener("click", async () => {
    switchInput.click();
});

function save_storage() {
    let filename = `chat ${new Date().toLocaleString()}.json`.replaceAll(":", "-");
    let data = {"options": {"g4f": ""}};
    for (let i = 0; i < appStorage.length; i++) {
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

document.getElementById("showLog").addEventListener("click", ()=> {
    log_storage.classList.remove("hidden");
    settings.classList.add("hidden");
    log_storage.scrollTop = log_storage.scrollHeight;
});