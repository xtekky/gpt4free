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
const chatPrompt        = document.getElementById("chatPrompt");
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
let usage_storage = {};
let reasoning_storage = {}
let is_demo = false;

messageInput.addEventListener("blur", () => {
    document.documentElement.scrollTop = 0;
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
            .replaceAll(/{"bucket_id":"([^"]+)"}/gm, (match, p1) => {
                size = parseInt(appStorage.getItem(`bucket:${p1}`), 10);
                return `**Bucket:** [[${p1}]](/backend-api/v2/files/${p1})${size ? ` (${formatFileSize(size)})` : ""}`;
            })
        )
            .replaceAll("<a href=", '<a target="_blank" href=')
            .replaceAll('<code>', '<code class="language-plaintext">')
    }
}

function render_reasoning(reasoning, final = false) {
    const inner_text = reasoning.text ? `<div class="reasoning_text${final ? " final hidden" : ""}">
        ${markdown_render(reasoning.text)}
    </div>` : "";
    return `<div class="reasoning_body">
        <div class="reasoning_title">
           <strong>Reasoning <i class="fa-solid fa-brain"></i>:</strong> ${escapeHtml(reasoning.status)}
        </div>
        ${inner_text}
    </div>`;
}

function filter_message(text) {
    return text.replaceAll(
        /<!-- generated images start -->[\s\S]+<!-- generated images end -->/gm, ""
    ).replace(/ \[aborted\]$/g, "").replace(/ \[error\]$/g, "");
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
if (window.hljs) {
    hljs.addPlugin(new HtmlRenderPlugin())
    hljs.addPlugin(new CopyButtonPlugin());
}
let typesetPromise = Promise.resolve();
const highlight = (container) => {
    if (window.hljs) {
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
    message_box.querySelectorAll(".message .content .provider").forEach(async (el) => {
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

    message_box.querySelectorAll(".message .fa-xmark").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const message_el = get_message_el(el);
                await remove_message(window.conversation_id, message_el.dataset.index);
                await safe_load_conversation(window.conversation_id, false);
            });
        }
    });

    message_box.querySelectorAll(".message .fa-clipboard").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                let message_el = get_message_el(el);
                let response = await fetch(message_el.dataset.object_url);
                let copyText = await response.text();
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

    message_box.querySelectorAll(".message .fa-file-export").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const elem = window.document.createElement('a');
                let filename = `chat ${new Date().toLocaleString()}.md`.replaceAll(":", "-");
                const conversation = await get_conversation(window.conversation_id);
                let buffer = "";
                conversation.items.forEach(message => {
                    buffer += `${message.role == 'user' ? 'User' : 'Assistant'}: ${message.content.trim()}\n\n\n`;
                });
                const file = new File([buffer.trim()], 'message.md', {type: 'text/plain'});
                const objectUrl = URL.createObjectURL(file);
                elem.href = objectUrl;
                elem.download = filename;        
                document.body.appendChild(elem);
                elem.click();        
                document.body.removeChild(elem);
                el.classList.add("clicked");
                setTimeout(() => el.classList.remove("clicked"), 1000);
                URL.revokeObjectURL(objectUrl);
            })
        }
    });

    message_box.querySelectorAll(".message .fa-volume-high").forEach(async (el) => {
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

    message_box.querySelectorAll(".message .regenerate_button").forEach(async (el) => {
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

    message_box.querySelectorAll(".message .continue_button").forEach(async (el) => {
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

    message_box.querySelectorAll(".message .fa-whatsapp").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                const text = get_message_el(el).innerText;
                window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank');
            });
        }
    });

    message_box.querySelectorAll(".message .fa-print").forEach(async (el) => {
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

    message_box.querySelectorAll(".message .reasoning_title").forEach(async (el) => {
        if (!("click" in el.dataset)) {
            el.dataset.click = "true";
            el.addEventListener("click", async () => {
                let text_el = el.parentElement.querySelector(".reasoning_text");
                if (text_el) {
                    text_el.classList[text_el.classList.contains("hidden") ? "remove" : "add"]("hidden");
                }
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

const handle_ask = async (do_ask_gpt = true) => {
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

    let message_index = await add_message(window.conversation_id, "user", message);
    let message_id = get_message_id();

    let images = [];
    if (do_ask_gpt) {
        if (imageInput.dataset.objects) {
            imageInput.dataset.objects.split(" ").forEach((object)=>URL.revokeObjectURL(object))
            delete imageInput.dataset.objects;
        }
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput
        if (input.files.length > 0) {
            for (const file of input.files) {
                images.push(URL.createObjectURL(file));
            }
            imageInput.dataset.objects = images.join(" ");
        }
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
                </div>
            </div>
        </div>
    `;
    highlight(message_box);
    if (do_ask_gpt) {
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
    } else {
        await lazy_scroll_to_bottom();
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
    regenerate_button.classList.remove("regenerate-hidden");
    stop_generating.classList.add("stop_generating-hidden");
    let key;
    for (key in controller_storage) {
        if (!controller_storage[key].signal.aborted) {
            console.log(`aborted ${window.conversation_id} #${key}`);
            try {
                controller_storage[key].abort();
            } finally {
                let message = message_storage[key];
                if (message) {
                    content_storage[key].inner.innerHTML += " [aborted]";
                    message_storage[key] += " [aborted]";
                }
            }
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

const prepare_messages = (messages, message_index = -1, do_continue = false, do_filter = true) => {
    messages = [ ...messages ]
    if (message_index != null) {
        console.debug("Messages Index:", message_index);

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
            console.debug("Messages filtered:", messages);
        }
    }
    // Combine assistant messages
    // let last_message;
    // let new_messages = [];
    // messages.forEach((message) => {
    //     message_copy = { ...message };
    //     if (last_message) {
    //         if (last_message["role"] == message["role"] &&  message["role"] == "assistant") {
    //             message_copy["content"] = last_message["content"] + message_copy["content"];
    //             new_messages.pop();
    //         }
    //     }
    //     last_message = message_copy;
    //     new_messages.push(last_message);
    // });
    // messages = new_messages;
    // console.log(2, messages);

    // Insert system prompt as first message
    let final_messages = [];
    if (chatPrompt?.value) {
        final_messages.push({
            "role": "system",
            "content": chatPrompt.value
        });
    }

    // Remove history, only add new user messages
    let filtered_messages = [];
    // The message_index is null on count total tokens
    if (document.getElementById('history')?.checked && do_filter && message_index != null) {
        while (last_message = messages.pop()) {
            if (last_message["role"] == "user") {
                filtered_messages.push(last_message);
            } else {
                break;
            }
        }
        messages = filtered_messages.reverse();
        if (last_message) {
            console.debug("History removed:", messages)
        }
    }

    messages.forEach((new_message, i) => {
        // Copy message first
        new_message = { ...new_message };
        // Include last message, if do_continue
        if (i + 1 == messages.length && do_continue) {
            delete new_message.regenerate;
        }
        // Include only not regenerated messages
        if (new_message) {
            // Remove generated images from content
            if (new_message.content) {
                new_message.content = filter_message(new_message.content);
            }
            // Remove internal fields
            delete new_message.provider;
            delete new_message.synthesize;
            delete new_message.finish;
            delete new_message.usage;
            delete new_message.reasoning;
            delete new_message.conversation;
            delete new_message.continue;
            // Append message to new messages
            if (do_filter && !new_message.regenerate) {
                final_messages.push(new_message)
            } else if (!do_filter) {
                final_messages.push(new_message)
            }
        }
    });
    console.debug("Final messages:", final_messages)

    return final_messages;
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
                if (typeof value == "object" && value != null) {
                    value = JSON.stringify(value, null, 4);
                }
                if (saved_value) {
                    field_el.classList.add("saved");
                } else {
                    saved_value = value;
                }
                let placeholder;
                if (["api_key", "proof_token"].includes(key)) {
                    placeholder = saved_value && saved_value.length >= 22 ? (saved_value.substring(0, 12) + "*".repeat(12) + saved_value.substring(saved_value.length-12)) : value;
                } else {
                    placeholder = value == null ? "null" : value;
                }
                field_el.innerHTML = `<label for="${el_id}" title="">${key}:</label>`;
                if (Number.isInteger(value) && value != 1) {
                    max = value >= 4096 ? 8192 : 4096;
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${provider}[${key}]" value="${escapeHtml(value)}" class="slider" min="0" max="${max}" step="1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else if (typeof value == "number") {
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${provider}[${key}]" value="${escapeHtml(value)}" class="slider" min="0" max="2" step="0.1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else {
                    field_el.innerHTML += `<textarea id="${el_id}" name="${provider}[${key}]"></textarea>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                    input_el = field_el.querySelector("textarea");
                    if (value != null) {
                        input_el.dataset.text = value;
                    }
                    input_el.placeholder = placeholder;
                    if (!["api_key", "proof_token"].includes(key)) {
                        input_el.value = saved_value;
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
                            input_el.value = input_el.dataset.saved_value;
                        } else if (["api_key", "proof_token"].includes(key)) {
                            input_el.value = input_el.dataset.text;
                        }
                        input_el.style.height = (input_el.scrollHeight) + "px";
                    }
                    input_el.onblur = () => {
                        input_el.style.removeProperty("height");
                        if (["api_key", "proof_token"].includes(key)) {
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
                    input_el.value = input_el.dataset.text;
                }
                delete input_el.dataset.saved_value;
                appStorage.removeItem(el_id);
                field_el.classList.remove("saved");
            }
        });
        provider_forms.appendChild(form_el);
    }
}

async function add_message_chunk(message, message_id, provider, scroll, finish_message=null) {
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
        await api("log", {...message, provider: provider_storage[message_id]});
    } else if (message.type == "error") {
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
        error_storage[message_id] = message.message
        console.error(message.message);
        content_map.inner.innerHTML += markdown_render(`**An error occured:** ${message.message}`);
        let p = document.createElement("p");
        p.innerText = message.error;
        log_storage.appendChild(p);
        await api("log", {...message, provider: provider_storage[message_id]});
    } else if (message.type == "preview") {
        if (content_map.inner.clientHeight > 200)
            content_map.inner.style.height = content_map.inner.clientHeight + "px";
        if (img = content_map.inner.querySelector("img"))
            if (!img.complete)
                return;
        content_map.inner.innerHTML = markdown_render(message.preview);
    } else if (message.type == "content") {
        message_storage[message_id] += message.content;
        update_message(content_map, message_id, null, scroll);
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
        update_message(content_map, message_id, markdown_render(message.login), scroll);
    } else if (message.type == "finish") {
        finish_storage[message_id] = message.finish;
        if (finish_message) {
            await finish_message();
        }
    } else if (message.type == "usage") {
        usage_storage[message_id] = message.usage;
    } else if (message.type == "reasoning") {
        if (!reasoning_storage[message_id]) {
            reasoning_storage[message_id] = message;
            reasoning_storage[message_id].text = "";
        } else if (message.status) {
            reasoning_storage[message_id].status = message.status;
        } else if (message.token) {
            reasoning_storage[message_id].text += message.token;
        }
        update_message(content_map, message_id, render_reasoning(reasoning_storage[message_id]), scroll);
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

function is_stopped() {
    if (stop_generating.classList.contains('stop_generating-hidden')) {
        return true;
    }
    return false;
}

const ask_gpt = async (message_id, message_index = -1, regenerate = false, provider = null, model = null, action = null) => {
    if (!model && !provider) {
        model = get_selected_model()?.value || null;
        provider = providerSelect.options[providerSelect.selectedIndex]?.value;
    }
    let conversation = await get_conversation(window.conversation_id);
    if (!conversation) {
        return;
    }
    messages = prepare_messages(conversation.items, message_index, action=="continue");
    message_storage[message_id] = "";
    stop_generating.classList.remove("stop_generating-hidden");
    let scroll = true;
    if (message_index >= 0 && parseInt(message_index) + 1 < conversation.items.length) {
        scroll = false;
    }

    if (scroll) {
        await lazy_scroll_to_bottom();
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
        message_index: message_index,
    }
    if (scroll) {
        await lazy_scroll_to_bottom();
    }
    async function finish_message() {
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
        if (!error_storage[message_id] && message_storage[message_id]) {
            html = markdown_render(message_storage[message_id]);
            content_map.inner.innerHTML = html;
            highlight(content_map.inner);
            if (imageInput) imageInput.value = "";
            if (cameraInput) cameraInput.value = "";
        }
        if (message_storage[message_id]) {
            const message_provider = message_id in provider_storage ? provider_storage[message_id] : null;
            let usage = {};
            if (usage_storage[message_id]) {
                usage = usage_storage[message_id];
                delete usage_storage[message_id];
            }
            usage = {
                model: message_provider?.model,
                provider: message_provider?.name,
                ...usage
            }
            // Calculate usage if we don't have it jet
            if (document.getElementById("track_usage").checked && !usage.prompt_tokens && window.GPTTokenizer_cl100k_base) {
                const prompt_token_model = model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
                const prompt_tokens = GPTTokenizer_cl100k_base?.encodeChat(messages, prompt_token_model).length;
                const completion_tokens = count_tokens(message_provider?.model, message_storage[message_id]);
                usage = {
                    ...usage,
                    prompt_tokens: prompt_tokens,
                    completion_tokens: completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens
                }
            }
            // It is not regenerated, if it is the first response to a new question
            if (regenerate && message_index == -1) {
                let conversation = await get_conversation(window.conversation_id);
                regenerate = conversation.items[conversation.items.length-1]["role"] != "user";
            }
            // Create final message content
            const final_message = message_storage[message_id]
                                + (error_storage[message_id] ? " [error]" : "")
                                + (stop_generating.classList.contains('stop_generating-hidden') ? " [aborted]" : "")
            // Save message in local storage
            await add_message(
                window.conversation_id,
                "assistant",
                final_message,
                message_provider,
                message_index,
                synthesize_storage[message_id],
                regenerate,
                title_storage[message_id],
                finish_storage[message_id],
                usage,
                reasoning_storage[message_id],
                action=="continue"
            );
            delete message_storage[message_id];
            // Send usage to the server
            if (document.getElementById("track_usage").checked) {
                const user = localStorage.getItem("user");
                if (user) {
                    usage = {user: user, ...usage};
                }
                api("usage", usage);
            }
        }
        // Update controller storage
        if (controller_storage[message_id]) {
            delete controller_storage[message_id];
        }
        // Reload conversation if no error
        if (!error_storage[message_id]) {
            await safe_load_conversation(window.conversation_id, scroll);
        }
        let cursorDiv = message_el.querySelector(".cursor");
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);
        if (scroll) {
            await lazy_scroll_to_bottom();
        }
        await safe_remove_cancel_button();
        await register_message_buttons();
        await load_conversations();
        regenerate_button.classList.remove("regenerate-hidden");
    }
    try {
        let api_key;
        if (is_demo && provider == "Feature") {
            api_key = localStorage.getItem("user");
        } else if (is_demo && provider != "Custom") {
            api_key = localStorage.getItem("HuggingFace-api_key");
        } else {
            api_key = get_api_key_by_provider(provider);
        }
        if (is_demo && !api_key && provider != "Custom") {
            location.href = "/";
            return;
        }
        const input = imageInput && imageInput.files.length > 0 ? imageInput : cameraInput;
        const files = input && input.files.length > 0 ? input.files : null;
        const download_images = document.getElementById("download_images")?.checked;
        const api_base = provider == "Custom" ? document.getElementById(`${provider}-api_base`).value : null;
        const ignored = Array.from(settings.querySelectorAll("input.provider:not(:checked)")).map((el)=>el.value);
        await api("conversation", {
            id: message_id,
            conversation_id: window.conversation_id,
            conversation: provider && conversation.data && provider in conversation.data ? conversation.data[provider] : null,
            model: model,
            web_search: switchInput.checked,
            provider: provider,
            messages: messages,
            action: action,
            download_images: download_images,
            api_key: api_key,
            api_base: api_base,
            ignored: ignored,
        }, files, message_id, scroll, finish_message);
    } catch (e) {
        console.error(e);
        if (e.name != "AbortError") {
            error_storage[message_id] = true;
            content_map.inner.innerHTML += markdown_render(`**An error occured:** ${e}`);
        }
        await finish_message();
    }
};

async function scroll_to_bottom() {
    window.scrollTo(0, 0);
    message_box.scrollTop = message_box.scrollHeight;
}

async function lazy_scroll_to_bottom() {
    if (document.querySelector("#input-count input").checked) {
        await scroll_to_bottom();
    }
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
        add_url_to_history(`/chat/${conversation_id}`);
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
    if (chatPrompt) {
        chatPrompt.value = document.getElementById("systemPrompt")?.value;
    }
    load_conversations();
    hide_sidebar();
    say_hello();
};

function merge_messages(message1, message2) {
    let newContent = message2;
    // Remove start tokens
    if (newContent.startsWith("```")) {
        const index = newContent.indexOf("\n");
        if (index != -1) {
            newContent = newContent.substring(index);
        }
    } else if (newContent.startsWith("...")) {
        newContent = " " + newContent.substring(3);
    } else if (newContent.startsWith(message1)) {
        newContent = newContent.substring(message1.length);
    } else {
        // Remove duplicate lines
        let lines = message1.trim().split("\n");
        let lastLine = lines[lines.length - 1];
        let foundLastLine = newContent.indexOf(lastLine + "\n");
        if (foundLastLine != -1) {
            foundLastLine += 1;
        } else {
            foundLastLine = newContent.indexOf(lastLine);
        }
        if (foundLastLine != -1) {
            newContent = newContent.substring(foundLastLine + lastLine.length);
        } // Remove duplicate words
        else if (newContent.indexOf(" ") > 0) {
            let words = message1.trim().split(" ");
            let lastWord = words[words.length - 1];
            if (newContent.startsWith(lastWord)) {
                newContent = newContent.substring(lastWord.length);
            }
        }
    }
    return message1 + newContent;
}

// console.log(merge_messages("Hello", "Hello,\nhow are you?"));
// console.log(merge_messages("Hello", "Hello, how are you?"));
// console.log(merge_messages("Hello", "Hello,\nhow are you?"));
// console.log(merge_messages("Hello,\n", "Hello,\nhow are you?"));
// console.log(merge_messages("Hello,\n", "how are you?"));
// console.log(merge_messages("1 != 2", "1 != 2;"));
// console.log(merge_messages("1 != 2", "```python\n1 != 2;"));
// console.log(merge_messages("1 != 2;\n1 != 3;\n", "1 != 2;\n1 != 3;\n"));

const load_conversation = async (conversation_id, scroll=true) => {
    let conversation = await get_conversation(conversation_id);
    let messages = conversation?.items || [];
    console.debug("Conversation:", conversation)

    if (!conversation) {
        return;
    }
    let title = conversation.title || conversation.new_title;
    title = title ? `${title} - g4f` : window.title;
    if (title) {
        document.title = title;
    }

    if (chatPrompt) {
        chatPrompt.value = conversation.system || "";
    }

    let elements = [];
    let last_model = null;
    let providers = [];
    let buffer = "";
    let completion_tokens = 0;

    messages.forEach((item, i) => {
        if (item.continue) {
            elements.pop();
        } else {
            buffer = "";
        }
        buffer = buffer.replace(/ \[aborted\]$/g, "").replace(/ \[error\]$/g, "");
        new_content = item.content.replace(/ \[aborted\]$/g, "").replace(/ \[error\]$/g, "");
        buffer = merge_messages(buffer, new_content);
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

        const file = new File([buffer], 'message.md', {type: 'text/plain'});
        const objectUrl = URL.createObjectURL(file);

        let add_buttons = [];
        // Find buttons to add
        actions = ["variant"]
        // Add continue button if possible
        if (item.role == "assistant") {
            let reason = "stop";
            // Read finish reason from conversation
            if (item.finish && item.finish.reason) {
                reason = item.finish.reason;
            }
            let lines = buffer.trim().split("\n");
            let lastLine = lines[lines.length - 1];
            // Has a stop or error token at the end
            if (lastLine.endsWith("[aborted]") || lastLine.endsWith("[error]")) {
                reason = "error";
            // Has an even number of start or end code tags
            } else if (reason == "stop" && buffer.split("```").length - 1 % 2 === 1) {
                reason = "length";
            }
            if (reason == "length" || reason == "max_tokens" || reason == "error") {
                actions.push("continue")
            }
        }

        add_buttons.push(`<button class="options_button">
            <div>
                <span><i class="fa-brands fa-whatsapp"></i></span>
                <span><i class="fa-solid fa-volume-high"></i></i></span>
                <span><i class="fa-solid fa-print"></i></span>
                <span><i class="fa-solid fa-file-export"></i></span>
                <span><i class="fa-regular fa-clipboard"></i></span>
            </div>
            <i class="fa-solid fa-plus"></i>
        </button>`);

        if (actions.includes("variant")) {
            add_buttons.push(`<button class="regenerate_button">
                <span>Regenerate</span>
                <i class="fa-solid fa-rotate"></i>
            </button>`);
        }
        if (actions.includes("continue")) {
            if (messages.length >= i - 1) {
                add_buttons.push(`<button class="continue_button">
                    <span>Continue</span>
                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                </button>`);
            }
        }

        if (!item.continue) {
            completion_tokens = 0;
        }
        completion_tokens += item.usage?.completion_tokens ? item.usage.completion_tokens : 0;
        let next_usage = messages.length > next_i ? messages[next_i].usage : null;
        let prompt_tokens = next_usage?.prompt_tokens ? next_usage?.prompt_tokens : 0

        elements.push(`
            <div class="message${item.regenerate ? " regenerate": ""}" data-index="${i}" data-object_url="${objectUrl}" data-synthesize_url="${synthesize_url}">
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
                    <div class="content_inner">
                        ${item.reasoning ? render_reasoning(item.reasoning, true): ""}
                        ${markdown_render(buffer)}
                    </div>
                    <div class="count">
                        ${count_words_and_tokens(buffer, next_provider?.model, completion_tokens, prompt_tokens)}
                        ${add_buttons.join("")}
                    </div>
                </div>
            </div>
        `);
    });

    if (window.GPTTokenizer_cl100k_base) {
        const filtered = prepare_messages(messages, null, true, false);
        if (filtered.length > 0) {
            last_model = last_model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
            let count_total = GPTTokenizer_cl100k_base?.encodeChat(filtered, last_model).length
            if (count_total > 0) {
                elements.push(`<div class="count_total">(${count_total} total tokens)</div>`);
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

    if (scroll && document.querySelector("#input-count input").checked) {
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
            system: chatPrompt?.value,
            items: [],
        });
    }
    try {
        add_url_to_history(`/chat/${conversation_id}`);
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
        conversation.system = chatPrompt?.value;
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
    usage = null,
    reasoning = null,
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
    if (usage) {
        new_message.usage = usage;
    }
    if (reasoning) {
        new_message.reasoning = reasoning;
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
                    <i onclick="delete_conversation('${conversation.id}')" class="fa-solid fa-trash"></i>
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
    await hide_settings();
    if (window.location.pathname.endsWith("/menu/") || window.location.pathname.endsWith("/settings/")) {
        history.back();
    }
}

async function hide_settings() {
    settings.classList.add("hidden");
    let provider_forms = document.querySelectorAll(".provider_forms from");
    Array.from(provider_forms).forEach((form) => form.classList.add("hidden"));
}

window.addEventListener('popstate', hide_sidebar, false);

sidebar_button.addEventListener("click", async () => {
    if (sidebar.classList.contains("shown")) {
        await hide_sidebar();
    } else {
        await show_menu();
    }
    window.scrollTo(0, 0);
});

function add_url_to_history(url) {
    if (!window?.pywebview) {
        history.pushState({}, null, url);
    }
}

async function show_menu() {
    sidebar.classList.add("shown");
    sidebar_button.classList.add("rotated");
    await hide_settings();
    add_url_to_history("/chat/menu/");
}

function open_settings() {
    if (settings.classList.contains("hidden")) {
        chat.classList.add("hidden");
        sidebar.classList.remove("shown");
        settings.classList.remove("hidden");
        add_url_to_history("/chat/settings/");
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
                        appStorage.setItem(element.id, element.value);
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
        if (element.id.endsWith("-api_key")) {
            element.addEventListener('focus', async (event) => {
                if (element.dataset.value) {
                    element.value = element.dataset.value
                }
            });
            element.addEventListener('blur', async (event) => {
                element.dataset.value = element.value;
                if (element.value) {
                    element.placeholder = element.value && element.value.length >= 22 ? (element.value.substring(0, 12)+"*".repeat(12)+element.value.substring(element.value.length-12)) : "*".repeat(element.value.length);
                } else if (element.placeholder != "api_key") {
                    element.placeholder = "";
                }
                element.value = ""
            });
        }
    });
}

const load_settings_storage = async () => {
    const optionElements = document.querySelectorAll(optionElementsSelector);
    optionElements.forEach((element) => {
        if (element.name && element.name != element.id && (value = appStorage.getItem(element.name))) {
            appStorage.setItem(element.id, value);
            appStorage.removeItem(element.name);
        }
        if (!(value = appStorage.getItem(element.id))) {
            return;
        }
        if (value) {
            switch (element.type) {
                case "checkbox":
                    element.checked = value === "true";
                    break;
                case "select-one":
                    element.value = value;
                    break;
                case "text":
                case "number":
                case "textarea":
                    if (element.id.endsWith("-api_key")) {
                        element.placeholder = value && value.length >= 22 ? (value.substring(0, 12)+"*".repeat(12)+value.substring(value.length-12)) : "*".repeat(value.length);
                        element.dataset.value = value;
                    } else {
                        element.value = value;
                    }
                    break;
                default:
                    console.warn("`Unresolved element type:", element.type);
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

function count_tokens(model, text, prompt_tokens = 0) {
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
    if (window.GPTTokenizer_cl100k_base && window.GPTTokenizer_o200k_base) {
        if (model?.startsWith("gpt-4o") || model?.startsWith("o1")) {
            return GPTTokenizer_o200k_base?.encode(text, model).length;
        } else {
            model = model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
            return GPTTokenizer_cl100k_base?.encode(text, model).length;
        }
    } else {
        return prompt_tokens;
    }
}

function count_words(text) {
    return text.trim().match(/[\w\u4E00-\u9FA5]+/gu)?.length || 0;
}

function count_chars(text) {
    return text.match(/[^\s\p{P}]/gu)?.length || 0;
}

function count_words_and_tokens(text, model, completion_tokens, prompt_tokens) {
    text = filter_message(text);
    return `(${count_words(text)} words, ${count_chars(text)} chars, ${completion_tokens ? completion_tokens : count_tokens(model, text, prompt_tokens)} tokens)`;
}

function update_message(content_map, message_id, content = null, scroll = true) {
    content_map.update_timeouts.push(setTimeout(() => {
        if (!content) {
            if (reasoning_storage[message_id]) {
                content = render_reasoning(reasoning_storage[message_id], true) + markdown_render(message_storage[message_id]);
            } else {
                content = markdown_render(message_storage[message_id]);
            }
            let lastElement, lastIndex = null;
            for (element of ['</p>', '</code></pre>', '</p>\n</li>\n</ol>', '</li>\n</ol>', '</li>\n</ul>']) {
                const index = content.lastIndexOf(element)
                if (index - element.length > lastIndex) {
                    lastElement = element;
                    lastIndex = index;
                }
            }
            if (lastIndex) {
                content = content.substring(0, lastIndex) + '<span class="cursor"></span>' + lastElement;
            }
        }
        if (error_storage[message_id]) {
            content_map.inner.innerHTML = message + markdown_render(`**An error occured:** ${error_storage[message_id]}`);
        } else {
            content_map.inner.innerHTML = content;
        }
        content_map.count.innerText = count_words_and_tokens(message_storage[message_id], provider_storage[message_id]?.model);
        highlight(content_map.inner);
        if (scroll) {
            lazy_scroll_to_bottom();
        }
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
    }, 100));
};

let countFocus = messageInput;
const count_input = async () => {
    if (countFocus.value) {
        if (window.matchMedia("(pointer:coarse)")) {
            inputCount.innerText = `(${count_tokens(get_selected_model()?.value, countFocus.value)} tokens)`;
        } else {
            inputCount.innerText = count_words_and_tokens(countFocus.value, get_selected_model()?.value);
        }
    } else {
        inputCount.innerText = "";
    }
};
messageInput.addEventListener("keyup", count_input);
chatPrompt.addEventListener("keyup", count_input);
chatPrompt.addEventListener("focus", function() {
    countFocus = chatPrompt;
    count_input();
});
chatPrompt.addEventListener("input", function() {
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
    } else {
        await on_api();
    }
});

window.addEventListener('pywebviewready', async function() {
    await on_api();
});

async function on_load() {
    count_input();
    if (/\/settings\//.test(window.location.href)) {
        open_settings();
    } else if (/\/chat\/[^?]+/.test(window.location.href)) {
        load_conversation(window.conversation_id);
    } else {
        chatPrompt.value = document.getElementById("systemPrompt")?.value || "";
        let chat_url = new URL(window.location.href)
        let chat_params = new URLSearchParams(chat_url.search);
        if (chat_params.get("prompt")) {
            messageInput.value = `${chat_params.title}\n${chat_params.prompt}\n${chat_params.url}`.trim();
            messageInput.style.height = messageInput.scrollHeight  + "px";
            messageInput.focus();
            //await handle_ask();
        } else {
            say_hello()
        }
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
    load_version();
    let prompt_lock = false;
    messageInput.addEventListener("keydown", async (evt) => {
        if (prompt_lock) return;
        // If not mobile and not shift enter
        let do_enter = messageInput.value.endsWith("\n\n");
        if (do_enter || !window.matchMedia("(pointer:coarse)").matches && evt.keyCode === 13 && !evt.shiftKey) {
            evt.preventDefault();
            console.log("pressed enter");
            prompt_lock = true;
            setTimeout(()=>prompt_lock=false, 3000);
            await handle_ask();
        } else {
            messageInput.style.height = messageInput.scrollHeight  + "px";
        }
    });
    sendButton.querySelector(".fa-paper-plane").addEventListener(`click`, async () => {
        console.log("clicked send");
        if (prompt_lock) return;
        prompt_lock = true;
        setTimeout(()=>prompt_lock=false, 3000);
        stop_recognition();
        await handle_ask();
    });
    sendButton.querySelector(".fa-square-plus").addEventListener(`click`, async () => {
        stop_recognition();
        await handle_ask(false);
    });
    messageInput.addEventListener(`click`, async () => {
        stop_recognition();
    });

    let provider_options = [];
    models = await api("models");
    models.forEach((model) => {
        let option = document.createElement("option");
        option.value = model.name;
        option.text = model.name + (model.image ? " (Image Generation)" : "") + (model.vision ? " (Image Upload)" : "");
        option.dataset.providers = model.providers.join(" ");
        modelSelect.appendChild(option);
        is_demo = model.demo;
    });
    let login_urls;
    if (is_demo) {
        if (!localStorage.getItem("user")) {
            location.href = "/";
            return;
        }
        providerSelect.innerHTML = `
            <option value="">Demo Mode</option>
            <option value="Feature">Feature Provider</option>
            <option value="Custom">Custom Provider</option>`;
        providerSelect.selectedIndex = 0;
        document.getElementById("pin").disabled = true;
        document.getElementById("refine")?.parentElement.classList.add("hidden")
        const track_usage = document.getElementById("track_usage");
        track_usage.checked = true;
        track_usage.disabled = true;
        Array.from(modelSelect.querySelectorAll(':not([data-providers])')).forEach((option)=>{
            if (!option.disabled && option.value) {
                option.remove();
            }
        });
        login_urls = {
            "Custom": ["Custom Provider", "", []],
            "HuggingFace": ["HuggingFace", "", []],
        };
    } else {
        providers = await api("providers")
        providers.sort((a, b) => a.label.localeCompare(b.label));
        login_urls = {};
        providers.forEach((provider) => {
            let option = document.createElement("option");
            option.value = provider.name;
            option.dataset.label = provider.label;
            option.text = provider.label
                + (provider.vision ? " (Image Upload)" : "")
                + (provider.image ? " (Image Generation)" : "")
                + (provider.nodriver ? " (Browser)" : "")
                + (!provider.nodriver && provider.auth ? " (Auth)" : "");
            if (provider.parent)
                option.dataset.parent = provider.parent;
            providerSelect.appendChild(option);

            if (provider.parent) {
                if (!login_urls[provider.parent]) {
                    login_urls[provider.parent] = [provider.label, provider.login_url, [provider.name]];
                } else {
                    login_urls[provider.parent][2].push(provider.name);
                }
            } else if (provider.login_url) {
                if (!login_urls[provider.name]) {
                    login_urls[provider.name] = [provider.label, provider.login_url, []];
                } else {
                    login_urls[provider.name][0] = provider.label;
                    login_urls[provider.name][1] = provider.login_url;
                }
            }
        });

        let providersContainer = document.createElement("div");
        providersContainer.classList.add("field", "collapsible");
        providersContainer.innerHTML = `
            <div class="collapsible-header">
                <span class="label">Providers (Enable/Disable)</span>
                <i class="fa-solid fa-chevron-down"></i>
            </div>
            <div class="collapsible-content hidden"></div>
        `;
        settings.querySelector(".paper").appendChild(providersContainer);

        providers.forEach((provider) => {
            if (!provider.parent) {
                let option = document.createElement("div");
                option.classList.add("provider-item");
                option.innerHTML = `
                    <span class="label">Enable ${provider.label}</span>
                    <input id="Provider${provider.name}" type="checkbox" name="Provider${provider.name}" value="${provider.name}" class="provider" checked="">
                    <label for="Provider${provider.name}" class="toogle" title="Remove provider from dropdown"></label>
                `;
                option.querySelector("input").addEventListener("change", (event) => load_provider_option(event.target, provider.name));
                providersContainer.querySelector(".collapsible-content").appendChild(option);
                provider_options[provider.name] = option;
            }
        });

        providersContainer.querySelector(".collapsible-header").addEventListener('click', (e) => {
            providersContainer.querySelector(".collapsible-content").classList.toggle('hidden');
            providersContainer.querySelector(".collapsible-header").classList.toggle('active');
        });
    }

    if (appStorage.getItem("provider")) {
        await load_provider_models(appStorage.getItem("provider"))
    } else {
        providerSelect.selectedIndex = 0;
    }

    let providersListContainer = document.createElement("div");
    providersListContainer.classList.add("field", "collapsible");
    providersListContainer.innerHTML = `
        <div class="collapsible-header">
            <span class="label">Providers API key</span>
            <i class="fa-solid fa-chevron-down"></i>
        </div>
        <div class="collapsible-content hidden"></div>
    `;
    settings.querySelector(".paper").appendChild(providersListContainer);

    for (let [name, [label, login_url, childs]] of Object.entries(login_urls)) {
        if (!login_url && !is_demo) {
            continue;
        }
        let providerBox = document.createElement("div");
        providerBox.classList.add("field", "box");
        childs = childs.map((child) => `${child}-api_key`).join(" ");
        providerBox.innerHTML = `
            <label for="${name}-api_key" class="label" title="">${label}:</label>
            <input type="text" id="${name}-api_key" name="${name}[api_key]" class="${childs}" placeholder="api_key" autocomplete="off"/>
        ` + (login_url ? `<a href="${login_url}" target="_blank" title="Login to ${label}">Get API key</a>` : "");
        providersListContainer.querySelector(".collapsible-content").appendChild(providerBox);
    }

    providersListContainer.querySelector(".collapsible-header").addEventListener('click', (e) => {
        providersListContainer.querySelector(".collapsible-content").classList.toggle('hidden');
        providersListContainer.querySelector(".collapsible-header").classList.toggle('active');
    });

    register_settings_storage();
    await load_settings_storage();
    Object.entries(provider_options).forEach(
        ([provider_name, option]) => load_provider_option(option.querySelector("input"), provider_name)
    );

    const hide_systemPrompt = document.getElementById("hide-systemPrompt")
    const slide_systemPrompt_icon = document.querySelector(".slide-systemPrompt i");
    if (hide_systemPrompt.checked) {
        chatPrompt.classList.add("hidden");
        slide_systemPrompt_icon.classList.remove("fa-angles-up");
        slide_systemPrompt_icon.classList.add("fa-angles-down");
    }
    hide_systemPrompt.addEventListener('change', async (event) => {
        if (event.target.checked) {
            chatPrompt.classList.add("hidden");
        } else {
            chatPrompt.classList.remove("hidden");
        }
    });
    document.querySelector(".slide-systemPrompt")?.addEventListener("click", () => {
        hide_systemPrompt.click();
        const checked = hide_systemPrompt.checked;
        chatPrompt.classList[checked ? "add": "remove"]("hidden");
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
    document.getElementById('recognition-language').placeholder = get_navigator_language();
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
        new_version.innerHTML = `G4F ${link}&nbsp;&nbsp;🆕`;
        new_version.addEventListener("click", ()=>new_version.parentElement.removeChild(new_version));
        document.body.appendChild(new_version);
    } else {
        text += versions["version"];
    }
    document.getElementById("version_text").innerHTML = text
    setTimeout(load_version, 1000 * 60 * 60); // 1 hour
}

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
});

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

function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    while (bytes >= 1024 && unitIndex < units.length - 1) {
        bytes /= 1024;
        unitIndex++;
    }
    return `${bytes.toFixed(2)} ${units[unitIndex]}`;
}

async function upload_files(fileInput) {
    const paperclip = document.querySelector(".user-input .fa-paperclip");
    const bucket_id = uuid();
    paperclip.classList.add("blink");

    const formData = new FormData();
    Array.from(fileInput.files).forEach(file => {
        formData.append('files[]', file);
    });
    await fetch("/backend-api/v2/files/" + bucket_id, {
        method: 'POST',
        body: formData
    });

    let do_refine = document.getElementById("refine")?.checked;
    function connectToSSE(url) {
        const eventSource = new EventSource(url);
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.error) {
                inputCount.innerText = `Error: ${data.error.message}`;
                paperclip.classList.remove("blink");
                fileInput.value = "";
            } else if (data.action == "load") {
                inputCount.innerText = `Read data: ${formatFileSize(data.size)}`;
            } else if (data.action == "refine") {
                inputCount.innerText = `Refine data: ${formatFileSize(data.size)}`;
            } else if (data.action == "download") {
                inputCount.innerText = `Download: ${data.count} files`;
            } else if (data.action == "done") {
                if (do_refine) {
                    do_refine = false;
                    connectToSSE(`/backend-api/v2/files/${bucket_id}?refine_chunks_with_spacy=true`);
                    return;
                }
                appStorage.setItem(`bucket:${bucket_id}`, data.size);
                inputCount.innerText = "Files are loaded successfully";
                if (!messageInput.value) {
                    messageInput.value = JSON.stringify({bucket_id: bucket_id});
                    handle_ask(false);
                } else {
                    messageInput.value += (messageInput.value ? "\n" : "") + JSON.stringify({bucket_id: bucket_id}) + "\n";
                    paperclip.classList.remove("blink");
                    fileInput.value = "";
                }
            }
        };
        eventSource.onerror = (event) => {
            eventSource.close();
            paperclip.classList.remove("blink");
        }
    }
    connectToSSE(`/backend-api/v2/files/${bucket_id}`);
}

fileInput.addEventListener('change', async (event) => {
    if (fileInput.files.length) {
        type = fileInput.files[0].name.split('.').pop()
        if (type == "har") {
            return await upload_cookies();
        } else if (type != "json") {
            await upload_files(fileInput);
        }
        fileInput.dataset.type = type
        if (type == "json") {
            const reader = new FileReader();
            reader.addEventListener('load', async (event) => {
                const data = JSON.parse(event.target.result);
                if (data.options && "g4f" in data.options) {
                    let count = 0;
                    Object.keys(data).forEach(key => {
                        if (key != "options" && !localStorage.getItem(key)) {
                            appStorage.setItem(key, JSON.stringify(data[key]));
                            count += 1;
                        }
                    });
                    await load_conversations();
                    fileInput.value = "";
                    inputCount.innerText = `${count} Conversations were imported successfully`;
                } else {
                    is_cookie_file = data.api_key;
                    if (Array.isArray(data)) {
                        data.forEach((item) => {
                            if (item.domain && item.name && item.value) {
                                is_cookie_file = true;
                            }
                        });
                    }
                    if (is_cookie_file) {
                        await upload_cookies();
                    } else {
                        await upload_files(fileInput);
                    }
                }
            });
            reader.readAsText(fileInput.files[0]);
        }
    }
});

if (!window.matchMedia("(pointer:coarse)").matches) {
    document.getElementById("image").setAttribute("multiple", "multiple");
}

chatPrompt?.addEventListener("input", async () => {
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

async function api(ressource, args=null, files=null, message_id=null, scroll=true, finish_message=null) {
    if (window?.pywebview) {
        if (args !== null) {
            if (ressource == "conversation") {
                return pywebview.api[`get_${ressource}`](args, message_id, scroll);
            }
            if (ressource == "models") {
                ressource = "provider_models";
            }
            return pywebview.api[`get_${ressource}`](args);
        }
        return pywebview.api[`get_${ressource}`]();
    }
    const headers = {};
    let url = `/backend-api/v2/${ressource}`;
    let response;
    if (ressource == "models" && args) {
        api_key = get_api_key_by_provider(args);
        if (api_key) {
            headers.x_api_key = api_key;
        }
        api_base = args == "Custom" ? document.getElementById(`${args}-api_base`).value : null;
        if (api_base) {
            headers.x_api_base = api_base;
        }
        url = `/backend-api/v2/${ressource}/${args}`;
    } else if (ressource == "conversation") {
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
        // On Ratelimit
        if (response.status == 429) {
            // They are still pending requests?
            for (let key in controller_storage) {
                if (!controller_storage[key].signal.aborted) {
                    console.error(response);
                    await finish_message();
                    return;
                }
            }
            setTimeout(async () => {
                response = await fetch(url, {
                    method: 'POST',
                    signal: controller_storage[message_id].signal,
                    headers: headers,
                    body: body,
                });
                if (response.status != 200) {
                    console.error(response);
                }
                await read_response(response, message_id, args.provider || null, scroll, finish_message);
                await finish_message();
            }, 20000) // Wait 20 secounds on rate limit
        } else {
            await read_response(response, message_id, args.provider || null, scroll, finish_message);
            await finish_message();
            return;
        }
    } else if (args) {
        if (ressource == "log") {
            if (!document.getElementById("report_error").checked) {
                return;
            }
            url = `https://roxky-g4f-demo.hf.space${url}`;
        }
        headers['content-type'] = 'application/json';
        response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(args),
        });
    }
    if (!response) {
        response = await fetch(url, {headers: headers});
    }
    if (response.status != 200) {
        console.error(response);
    }
    return await response.json();
}

async function read_response(response, message_id, provider, scroll, finish_message) {
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
                add_message_chunk(JSON.parse(buffer + line), message_id, provider, scroll, finish_message);
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
        api_key = document.querySelector(`.${provider}-api_key`)?.id || null;
        if (api_key == null) {
            api_key = document.getElementById(`${provider}-api_key`)?.id || null;
        }
        if (api_key) {
            api_key = appStorage.getItem(api_key);
        }
    }
    return api_key;
}

async function load_provider_models(provider=null) {
    if (!provider) {
        provider = providerSelect.value;
    }
    modelProvider.innerHTML = '';
    modelProvider.name = `model[${provider}]`;
    if (!provider) {
        modelProvider.classList.add("hidden");
        modelSelect.classList.remove("hidden");
        return;
    }
    const models = await api('models', provider);
    if (models && models.length > 0) {
        modelSelect.classList.add("hidden");
        modelProvider.classList.remove("hidden");
        let defaultIndex = 0;
        models.forEach((model, i) => {
            let option = document.createElement('option');
            option.value = model.model;
            option.dataset.label = model.model;
            option.text = `${model.model}${model.image ? " (Image Generation)" : ""}${model.vision ? " (Image Upload)" : ""}`;
            modelProvider.appendChild(option);
            if (model.default) {
                defaultIndex = i;
            }
        });
        modelProvider.selectedIndex = defaultIndex;
        let value = appStorage.getItem(modelProvider.name);
        if (value) {
            modelProvider.value = value;
        }
    } else {
        modelProvider.classList.add("hidden");
        modelSelect.classList.remove("hidden");
    }
};
providerSelect.addEventListener("change", () => {
    load_provider_models()
    messageInput.focus();
});
modelSelect.addEventListener("change", () => messageInput.focus());
modelProvider.addEventListener("change", () =>  messageInput.focus());

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
    messageInput.focus();
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

function import_memory() {
    if (!appStorage.getItem("mem0-api_key")) {
        return;
    }
    hide_sidebar();

    let count = 0;
    let user_id = appStorage.getItem("user") || appStorage.getItem("mem0-user_id");
    if (!user_id) {
        user_id = uuid();
        appStorage.setItem("mem0-user_id", user_id);
    }
    inputCount.innerText = `Start importing to Mem0...`;
    let conversations = [];
    for (let i = 0; i < appStorage.length; i++) {
        if (appStorage.key(i).startsWith("conversation:")) {
            let conversation = appStorage.getItem(appStorage.key(i));
            conversations.push(JSON.parse(conversation));
        }
    }
    conversations.sort((a, b) => (a.updated||0)-(b.updated||0));
    async function add_conversation_to_memory(i) {
        if (i > conversations.length - 1) {
            return;
        }
        let body = JSON.stringify(conversations[i]);
        response = await fetch(`/backend-api/v2/memory/${user_id}`, {
            method: 'POST',
            body: body,
            headers: {
                "content-type": "application/json",
                "x_api_key": appStorage.getItem("mem0-api_key")
            }
        });
        const result = await response.json();
        count += result.count;
        inputCount.innerText = `${count} Messages were imported`;
        add_conversation_to_memory(i + 1);
    }
    add_conversation_to_memory(0)
}

function get_navigator_language() {
    return navigator.languages.filter((v)=>v.includes("-"))[0] || navigator.language;
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let stop_recognition = ()=>{};
if (SpeechRecognition) {
    const mircoIcon = microLabel.querySelector("i");
    mircoIcon.classList.add("fa-microphone");
    mircoIcon.classList.remove("fa-microphone-slash");

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    let startValue;
    let buffer;
    let lastDebounceTranscript;
    recognition.onstart = function() {
        startValue = messageInput.value;
        lastDebounceTranscript = "";
        messageInput.readOnly = true;
        buffer = "";
    };
    recognition.onend = function() {
        messageInput.value = `${startValue ? startValue + "\n" : ""}${buffer}`;
        if (microLabel.classList.contains("recognition")) {
            recognition.start();
        } else {
            messageInput.readOnly = false;
            messageInput.focus();
        }
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
            inputCount.innerText = transcript;
            if (isFinal) {
                buffer = `${buffer ? buffer + "\n" : ""}${transcript.trim()}`;
            }
        }
    };

    stop_recognition = ()=>{
        if (microLabel.classList.contains("recognition")) {
            microLabel.classList.remove("recognition");
            recognition.stop();
            messageInput.value = `${startValue ? startValue + "\n" : ""}${buffer}`;
            count_input();
            return true;
        }
        return false;
    }

    microLabel.addEventListener("click", (e) => {
        if (!stop_recognition()) {
            microLabel.classList.add("recognition");
            const lang = document.getElementById("recognition-language")?.value;
            recognition.lang = lang || get_navigator_language();
            recognition.start();
        }
    });
}

document.getElementById("showLog").addEventListener("click", ()=> {
    log_storage.classList.remove("hidden");
    settings.classList.add("hidden");
    log_storage.scrollTop = log_storage.scrollHeight;
});
