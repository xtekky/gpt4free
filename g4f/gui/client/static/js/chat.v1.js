const colorThemes       = document.querySelectorAll('[name="theme"]');
const chatBody          = document.getElementById(`chatBody`);
const userInput         = document.getElementById("userInput");
const box_conversations = document.querySelector(`.top`);
const stop_generating   = document.querySelector(`.stop_generating`);
const regenerate_button = document.querySelector(`.regenerate`);
const sidebar           = document.querySelector(".sidebar");
const sidebar_buttons   = document.querySelectorAll(".mobile-sidebar-toggle");
const sendButton        = document.getElementById("sendButton");
const addButton         = document.getElementById("addButton");
const imageInput        = document.querySelector(".image-label");
const mediaSelect       = document.querySelector(".media-select");
const imageSelect       = document.getElementById("image");
const cameraInput       = document.getElementById("camera");
const fileInput         = document.getElementById("file");
const microLabel        = document.querySelector(".micro-label");
const inputCount        = document.getElementById("input-count").querySelector(".text");
const providerSelect    = document.getElementById("provider");
const modelSelect       = document.getElementById("model");
const modelProvider     = document.getElementById("model2");
const custom_model      = document.getElementById("model3");
const chatPrompt        = document.getElementById("chatPrompt");
const settings          = document.querySelector(".settings");
const chat              = document.querySelector(".chat-container");
const album             = document.querySelector(".images");
const log_storage       = document.querySelector(".log");
const switchInput       = document.getElementById("switch");
const searchButton      = document.getElementById("search");
const paperclip         = document.querySelector(".user-input .fa-paperclip");

const optionElementsSelector = ".settings input, .settings textarea, .chat-body input, #model, #model2, #provider";

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
let reasoning_storage = {};
let title_ids_storage = {};
let image_storage = {};
let is_demo = false;
let wakeLock = null;
let countTokensEnabled = true;
let reloadConversation = true;
let privateConversation = null;
let suggestions = null;

userInput.addEventListener("blur", () => {
    document.documentElement.scrollTop = 0;
});

userInput.addEventListener("focus", () => {
    document.documentElement.scrollTop = document.documentElement.scrollHeight;
});

appStorage = window.localStorage || {
    setItem: (key, value) => self[key] = value,
    getItem: (key) => self[key],
    removeItem: (key) => delete self[key],
    length: 0
}

let markdown_render = (content) => escapeHtml(content);
if (window.markdownit) {
    const markdown = window.markdownit();
    markdown_render = (content) => {
        if (Array.isArray(content)) {
            content = content.map((item) => {
                if (!item.name) {
                    size = parseInt(appStorage.getItem(`bucket:${item.bucket_id}`), 10);
                    return `**Bucket:** [[${item.bucket_id}]](${item.url})${size ? ` (${formatFileSize(size)})` : ""}`
                }
                if (item.name.endsWith(".wav") || item.name.endsWith(".mp3") || item.name.endsWith(".m4a")) {
                    return `<audio controls src="${item.url}"></audio>`;
                }
                if (item.name.endsWith(".mp4") || item.name.endsWith(".webm")) {
                    return `<video controls src="${item.url}"></video>`;
                }
                return `[![${item.name}](${item.url})]()`;
            }).join("\n");
        }
        content = content.replaceAll(/<!-- generated images start -->|<!-- generated images end -->/gm, "")
        return markdown.render(content)
            .replaceAll("<a href=", '<a target="_blank" href=')
            .replaceAll('<code>', '<code class="language-plaintext">')
            .replaceAll('&lt;i class=&quot;', '<i class="')
            .replaceAll('&quot;&gt;&lt;/i&gt;', '"></i>')
            .replaceAll('&lt;video controls src=&quot;', '<video loop autoplay controls muted src="')
            .replaceAll('&quot;&gt;&lt;/video&gt;', '"></video>')
            .replaceAll('&lt;audio controls src=&quot;', '<audio controls src="')
            .replaceAll('&quot;&gt;&lt;/audio&gt;', '"></audio>')
            .replaceAll('&lt;iframe type=&quot;text/html&quot; src=&quot;', '<iframe type="text/html" frameborder="0" allow="fullscreen" height="224" width="400" src="')
            .replaceAll('&quot;&gt;&lt;/iframe&gt;', `?enablejsapi=1"></iframe>`)
    }
}

function render_reasoning(reasoning, final = false) {
    const inner_text = reasoning.text ? `<div class="reasoning_text${final ? " final hidden" : ""}">
        ${markdown_render(reasoning.text)}
    </div>` : "";
    return `<div class="reasoning_body">
        <div class="reasoning_title">
           <strong>${reasoning.label ? reasoning.label :'Reasoning <i class="brain">🧠</i>'}: </strong>
           ${reasoning.status ? escapeHtml(reasoning.status) : '<i class="fas fa-spinner fa-spin"></i>'}
        </div>
        ${inner_text}
    </div>`;
}

function render_reasoning_text(reasoning) {
    return `${reasoning.label ? reasoning.label :'Reasoning 🧠'}: ${reasoning.status}\n\n${reasoning.text}\n\n`;
}

function filter_message(text) {
    if (Array.isArray(text)) {
        return text;
    }
    return filter_message_content(text.replaceAll(
        /<!-- generated images start -->[\s\S]+<!-- generated images end -->/gm, ""
    ))
}

function filter_message_content(text) {
    if (Array.isArray(text)) {
        return text;
    }
    return text.replace(/ \[aborted\]$/g, "").replace(/ \[error\]$/g, "")
}

function filter_message_image(text) {
    return text.replaceAll(
        /\]\(\/generate\//gm, "](/images/"
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
iframe_close.onclick = () => {
    iframe_container.classList.add("hidden");
    iframe.src = "";
}
iframe_container.appendChild(iframe_close);
document.body.appendChild(iframe_container);

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
    while(!(message_el.classList.contains('message')) && message_el.parentElement) {
        message_el = message_el.parentElement;
    }
    if (message_el.classList.contains('message')) {
        return message_el;
    }
}

function generateUUID() {
    if (crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = crypto.getRandomValues(new Uint8Array(1))[0] % 16;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function register_message_images() {
    chatBody.querySelectorAll(`.loading-indicator`).forEach((el) => el.remove());
    chatBody.querySelectorAll(`.message img:not([alt="your avatar"])`).forEach(async (el) => {
        if (!el.complete) {
            const indicator = document.createElement("span");
            indicator.classList.add("loading-indicator");
            indicator.innerHTML = `<i class="fas fa-spinner fa-spin"></i>`;
            el.parentElement.appendChild(indicator);
            el.onerror = () => {
                let indexCommand;
                if ((indexCommand = el.src.indexOf("/generate/")) >= 0) {
                    reloadConversation = false;
                    indexCommand = indexCommand + "/generate/".length + 1;
                    let newPath = el.src.substring(indexCommand)
                    let filename = newPath.replace(/(?:\?.+?|$)/, "");
                    let seed = Math.floor(Date.now() / 1000);
                    newPath = `https://image.pollinations.ai/prompt/${newPath}?seed=${seed}&nologo=true`;
                    let downloadUrl = newPath;
                    if (document.getElementById("download_media")?.checked) {
                        downloadUrl = `/images/${filename}?url=${escapeHtml(newPath)}`;
                    }
                    const link = document.createElement("a");
                    link.setAttribute("href", newPath);
                    const newImg = document.createElement("img");
                    newImg.src = downloadUrl;
                    newImg.alt = el.alt;
                    newImg.onload = () => {
                        lazy_scroll_to_bottom();
                        indicator.remove();
                    }
                    link.appendChild(newImg);
                    el.parentElement.appendChild(link);
                } else {
                    const span = document.createElement("span");
                    span.innerHTML = `<i class="fa-solid fa-plug"></i>${escapeHtml(el.alt)}`;
                    el.parentElement.appendChild(span);
                }
                el.remove();
                indicator.remove();
            }
            el.onload = () => {
                indicator.remove();
                lazy_scroll_to_bottom();
            }
        }
    });
}

const register_message_buttons = async () => {
    chatBody.querySelectorAll(".message .content .provider").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        const provider_link = el.querySelector("a");
        provider_link?.addEventListener("click", async (event) => {
            event.preventDefault();
            await load_provider_parameters(el.dataset.provider);
            const provider_forms = document.querySelector(".provider_forms");
            const provider_form = provider_forms.querySelector(`#${el.dataset.provider}-form`);
            if (provider_form) {
                provider_form.classList.remove("hidden");
                provider_forms.classList.remove("hidden");
                chat.classList.add("hidden");
            }
            return false;
        });
    });

    chatBody.querySelectorAll(".message .fa-xmark").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const message_el = get_message_el(el);
            if (message_el) {
                if ("index" in message_el.dataset) {
                    await remove_message(window.conversation_id, message_el.dataset.index);
                    chatBody.removeChild(message_el);
                }
            }
            reloadConversation = true;
            await safe_load_conversation(window.conversation_id, false);
        });
    });

    chatBody.querySelectorAll(".message .fa-clipboard").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
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
        });
    })

    chatBody.querySelectorAll(".message .fa-file-export").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const elem = window.document.createElement('a');
            let filename = `chat ${new Date().toLocaleString()}.txt`.replaceAll(":", "-");
            const conversation = await get_conversation(window.conversation_id);
            let buffer = "";
            conversation.items.forEach(message => {
                if (message.reasoning) {
                    buffer += render_reasoning_text(message.reasoning);
                }
                buffer += `${message.role == 'user' ? 'User' : 'Assistant'}: ${message.content.trim()}\n\n`;
            });
            var download = document.getElementById("download");
            download.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(buffer.trim()));
            download.setAttribute("download", filename);
            download.click();
            el.classList.add("clicked");
            setTimeout(() => el.classList.remove("clicked"), 1000);
        });
    })

    chatBody.querySelectorAll(".message .fa-volume-high").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const message_el = get_message_el(el);
            let audio;
            if (message_el.dataset.synthesize_url) {
                el.classList.add("active");
                setTimeout(()=>el.classList.remove("active"), 2000);
                const media_player = document.querySelector(".media-player");
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
    });

    chatBody.querySelectorAll(".message .regenerate_button").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const message_el = get_message_el(el);
            el.classList.add("clicked");
            setTimeout(() => el.classList.remove("clicked"), 1000);
            await ask_gpt(get_message_id(), message_el.dataset.index);
        });
    });

    chatBody.querySelectorAll(".message .continue_button").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            if (!el.disabled) {
                el.disabled = true;
                const message_el = get_message_el(el);
                el.classList.add("clicked");
                setTimeout(() => {el.classList.remove("clicked"); el.disabled = false}, 1000);
                await ask_gpt(get_message_id(), message_el.dataset.index, false, null, null, "continue");
            }
        });
    });

    chatBody.querySelectorAll(".message .fa-whatsapp").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const text = get_message_el(el).innerText;
            window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank');
            });
    });

    chatBody.querySelectorAll(".message .fa-print").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            const message_el = get_message_el(el);
            el.classList.add("clicked");
            chatBody.scrollTop = 0;
            message_el.classList.add("print");
            setTimeout(() => {
                el.classList.remove("clicked");
                message_el.classList.remove("print");
            }, 1000);
            window.print()
        });
    });

    chatBody.querySelectorAll(".message .fa-qrcode").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        const message_el = get_message_el(el);
        el.addEventListener("click", async () => {
            iframe.src = window.conversation_id ? `/qrcode/${window.conversation_id}#${message_el.dataset.index}` : '/qrcode';
            iframe_container.classList.remove("hidden");
        });
    });

    chatBody.querySelectorAll(".message .reasoning_title").forEach(async (el) => {
        if (el.dataset.click) {
            return
        }
        el.dataset.click = true;
        el.addEventListener("click", async () => {
            let text_el = el.parentElement.querySelector(".reasoning_text");
            if (text_el) {
                text_el.classList.toggle("hidden");
            }
        });
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

const handle_ask = async (do_ask_gpt = true, message = null) => {
    userInput.style.height = "82px";
    userInput.focus();
    await scroll_to_bottom();

    if (!message) {
        message = userInput.value.trim();
        if (!message) {
            return;
        }
        userInput.value = "";
        await count_input()
    }

    // Is message a url?
    const expression = /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/gi;
    const regex = new RegExp(expression);
    if (!Array.isArray(message) && message.match(regex)) {
        paperclip.classList.add("blink");
        const blob = new Blob([JSON.stringify([{url: message}])], { type: 'application/json' });
        const file = new File([blob], 'downloads.json', { type: 'application/json' }); // Create File object
        let formData = new FormData();
        formData.append('files', file); // Append as a file
        const bucket_id = generateUUID();
        await fetch(`/backend-api/v2/files/${bucket_id}`, {
            method: 'POST',
            body: formData
        });
        connectToSSE(`/backend-api/v2/files/${bucket_id}`, false, bucket_id); //Retrieve and refine
        return;
    }

    await add_conversation(window.conversation_id);
    let message_index = await add_message(window.conversation_id, "user", message);
    let message_id = get_message_id();

    const message_el = document.createElement("div");
    message_el.classList.add("message");
    message_el.dataset.index = message_index;
    message_el.innerHTML = `
        <div class="user">
            ${user_image}
            <i class="fa-solid fa-xmark"></i>
            <i class="fa-regular fa-phone-arrow-up-right"></i>
        </div>
        <div class="content"> 
            <div class="content_inner">
            ${markdown_render(message)}
            </div>
            <div class="count">
                ${countTokensEnabled ? count_words_and_tokens(message, get_selected_model()?.value) : ""}
            </div>
        </div>
    `;
    chatBody.appendChild(message_el);
    highlight(message_el);
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
        await safe_load_conversation(window.conversation_id, true);
        await load_conversations();
    }
};

async function safe_remove_cancel_button() {
    for (let key in controller_storage) {
        if (!controller_storage[key].signal.aborted) {
            return;
        }
    }
    stop_generating.classList.add("stop_generating-hidden");
    if (wakeLock) {
        wakeLock.release();
        wakeLock = null;
    }
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
    await safe_load_conversation(window.conversation_id, false);
});

document.querySelector(".media-player .fa-x").addEventListener("click", ()=>{
    const media_player = document.querySelector(".media-player");
    media_player.classList.remove("show");
    const audio = document.querySelector(".media-player audio");
    media_player.removeChild(audio);
});

document.getElementById("close_provider_forms").addEventListener("click", async () => {
    const provider_forms = document.querySelector(".provider_forms");
    provider_forms.classList.add("hidden");
    chat.classList.remove("hidden");
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
    // The message_index is null on count total tokens
    if (!do_continue && document.getElementById('history')?.checked && do_filter && message_index != null) {
        let filtered_messages = [];
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
        let old_form = document.getElementById(form_id);
        if (old_form) {
            old_form.remove();
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
                <input type="checkbox" id="${el_id}" name="${key}">
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
                if (Number.isInteger(value)) {
                    max = value == 42 || value >= 4096 ? 8192 : value >= 100 ? 4096 : value == 1 ? 10 : 100;
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${key}" value="${escapeHtml(value)}" class="slider" min="0" max="${max}" step="1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else if (typeof value == "number") {
                    field_el.innerHTML += `<input type="range" id="${el_id}" name="${key}" value="${escapeHtml(value)}" class="slider" min="0" max="2" step="0.1"/><output>${escapeHtml(value)}</output>`;
                    field_el.innerHTML += `<i class="fa-solid fa-xmark"></i>`;
                } else {
                    field_el.innerHTML += `<textarea id="${el_id}" name="${key}"></textarea>`;
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
                    input_el.nextElementSibling.value = input_el.value;
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
        await save_conversation(conversation_id, get_conversation_data(conversation));
    } else if (message.type == "auth") {
        error_storage[message_id] = message.message
        content_map.inner.innerHTML += markdown_render(`**An error occured:** ${message.message}`);
        let provider = provider_storage[message_id]?.name;
        let configEl = document.querySelector(`.settings .${provider}-api_key`);
        if (configEl) {
            configEl = configEl.parentElement.cloneNode(true);
            content_map.content.appendChild(configEl);
            await register_settings_storage();
        }
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
        if (finish_message) {
            await finish_message();
        }
        let p = document.createElement("p");
        p.innerText = message.error;
        log_storage.appendChild(p);
        await api("log", {...message, provider: provider_storage[message_id]});
    } else if (message.type == "preview") {
        if (img = content_map.inner.querySelector("img"))
            if (!img.complete)
                return;
            else
                img.src = message.images;
        else {
            content_map.inner.innerHTML = markdown_render(message.preview);
            await register_message_images();
        }
    } else if (message.type == "content") {
        message_storage[message_id] += message.content;
        update_message(content_map, message_id, null, scroll);
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
    } else if (message.type == "usage") {
        usage_storage[message_id] = message.usage;
    } else if (message.type == "reasoning") {
        if (!reasoning_storage[message_id]) {
            reasoning_storage[message_id] = message;
            reasoning_storage[message_id].text = message_storage[message_id];
            message_storage[message_id] = "";
        } else if (message.status) {
            reasoning_storage[message_id].status = message.status;
        } if (message.label) {
            reasoning_storage[message_id].label = message.label;
        } if (message.token) {
            reasoning_storage[message_id].text += message.token;
        }
        update_message(content_map, message_id, null, scroll);
    } else if (message.type == "parameters") {
        if (!parameters_storage[provider]) {
            parameters_storage[provider] = {};
        }
        Object.entries(message.parameters).forEach(([key, value]) => {
            parameters_storage[provider][key] = value;
        });
    } else if (message.type == "suggestions") {
        suggestions = message.suggestions;
    }
}

function is_stopped() {
    if (stop_generating.classList.contains('stop_generating-hidden')) {
        return true;
    }
    return false;
}

const requestWakeLock = async () => {
    try {
      wakeLock = await navigator.wakeLock.request('screen');
    }
    catch(err) {
      console.error(err);
    }
  };

const ask_gpt = async (message_id, message_index = -1, regenerate = false, provider = null, model = null, action = null) => {
    if (!model && !provider) {
        model = get_selected_model()?.value || null;
        provider = providerSelect.options[providerSelect.selectedIndex]?.value;
    }
    let conversation = await get_conversation(window.conversation_id);
    if (!conversation) {
        return;
    }
    await requestWakeLock();
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

    let suggestions_el = chatBody.querySelector('.suggestions');
    suggestions_el ? suggestions_el.remove() : null;
    if (countTokensEnabled) {
        let count_total = chatBody.querySelector('.count_total');
        count_total ? count_total.parentElement.removeChild(count_total) : null;
    }

    const message_el = document.createElement("div");
    message_el.classList.add("message");
    if (message_index != -1 || regenerate) {
        message_el.classList.add("regenerate");
    }
    message_el.innerHTML = `
        <div class="assistant">
            ${gpt_image}
            <i class="fa-solid fa-xmark"></i>
            <i class="fa-regular fa-phone-arrow-down-left"></i>
        </div>
        <div class="content">
            <div class="provider" data-provider="${provider}"></div>
            <div class="content_inner"><span class="cursor"></span></div>
            <div class="count"></div>
        </div>
    `;
    if (message_index == -1) {
        chatBody.appendChild(message_el);
    } else {
        parent_message = chatBody.querySelector(`.message[data-index="${message_index}"]`);
        if (!parent_message) {
            return;
        }
        parent_message.after(message_el);
    }

    controller_storage[message_id] = new AbortController();

    let content_el = message_el.querySelector('.content');
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
        }
        if (message_storage[message_id] || reasoning_storage[message_id]?.status) {
            const message_provider = message_id in provider_storage ? provider_storage[message_id] : null;
            let usage = {};
            if (usage_storage[message_id]) {
                usage = usage_storage[message_id];
                delete usage_storage[message_id];
            }
            // Calculate usage if we don't have it jet
            if (countTokensEnabled && document.getElementById("track_usage").checked && !usage.prompt_tokens && window.GPTTokenizer_cl100k_base) {
                const prompt_token_model = model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
                const prompt_tokens = GPTTokenizer_cl100k_base?.encodeChat(messages, prompt_token_model).length;
                const completion_tokens = count_tokens(message_provider?.model, message_storage[message_id])
                    + (reasoning_storage[message_id] ? count_tokens(message_provider?.model, reasoning_storage[message_id].text) : 0);
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
                filter_message_image(final_message),
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
                usage = {
                    model: message_provider?.model,
                    provider: message_provider?.name,
                    ...usage
                };
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
        if (!error_storage[message_id] && reloadConversation) {
            if(await safe_load_conversation(window.conversation_id, scroll)) {
                const new_message = Array.from(document.querySelectorAll(".message")).at(-1);
                const new_media = new_message.querySelector("audio, iframe");
                if (new_media) {
                    if (new_media.tagName == "IFRAME") {
                        if (YT) {
                            async function onPlayerReady(event) {
                                if (scroll) {
                                    await lazy_scroll_to_bottom();
                                }
                                event.target.setVolume(100);
                                event.target.playVideo();
                            }
                            player = new YT.Player(new_media, {
                                events: {
                                    'onReady': onPlayerReady,
                                }
                            });
                        }
                    } else {
                        new_media.play();
                    }
                }
            }
        }
        let cursorDiv = message_el.querySelector(".cursor");
        if (cursorDiv) cursorDiv.parentNode.removeChild(cursorDiv);
        if (scroll) {
            setTimeout(async () => {
                await lazy_scroll_to_bottom();
            }, 2000);
        }
        await safe_remove_cancel_button();
        await register_message_images();
        await register_message_buttons();
        await load_conversations();
        regenerate_button.classList.remove("regenerate-hidden");
    }
    try {
        let api_key;
        if (is_demo && !provider) {
            api_key = localStorage.getItem("HuggingFace-api_key");
        } else {
            api_key = get_api_key_by_provider(provider);
        }
        const download_media = document.getElementById("download_media")?.checked;
        let api_base;
        if (provider == "Custom") {
            api_base = document.getElementById("api_base")?.value;
            if (!api_base) {
                provider = "";
            }
        }
        const ignored = Array.from(settings.querySelectorAll("input.provider:not(:checked)")).map((el)=>el.value);
        let extra_parameters = [];
        for (el of document.getElementById(`${provider}-form`)?.querySelectorAll(".saved input, .saved textarea") || []) {
            let value = el.type == "checkbox" ? el.checked : el.value;
            try {
                value = await JSON.parse(value);
            } catch (e) {
            }
            extra_parameters[el.name] = value;
        };
        automaticOrientation = appStorage.getItem("automaticOrientation") != "false";
        aspect_ratio = automaticOrientation ? (window.innerHeight > window.innerWidth ? "9:16" : "16:9") : null;
        await api("conversation", {
            id: message_id,
            conversation_id: window.conversation_id,
            conversation: provider && conversation.data && provider in conversation.data ? conversation.data[provider] : null,
            model: model,
            web_search: switchInput.checked,
            provider: provider,
            messages: messages,
            action: action,
            download_media: download_media,
            api_key: api_key,
            api_base: api_base,
            ignored: ignored,
            aspect_ratio: aspect_ratio,
            ...extra_parameters
        }, Object.values(image_storage), message_id, scroll, finish_message);
    } catch (e) {
        console.error(e);
    }
};

async function scroll_to_bottom() {
    window.scrollTo(0, 0);
    chatBody.scrollTop = chatBody.scrollHeight;
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
    let messages = chatBody.getElementsByTagName(`div`);

    while (messages.length > 0) {
        chatBody.removeChild(messages[0]);
    }
};

var illegalRe = /[\/\?<>\\:\*\|":]/g;
var controlRe = /[\x00-\x1f\x80-\x9f]/g;
var reservedRe = /^\.+$/;
var windowsReservedRe = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$/i;

function sanitize(input, replacement) {
  var sanitized = input
    .replace(illegalRe, replacement)
    .replace(controlRe, replacement)
    .replace(reservedRe, replacement)
    .replace(windowsReservedRe, replacement);
  return sanitized.replaceAll(/\/|#|\s{2,}/g, replacement).trim();
}

async function set_conversation_title(conversation_id, title) {
    conversation = await get_conversation(conversation_id)
    conversation.new_title = title;
    const new_id = sanitize(title, " ");
    if (new_id && !appStorage.getItem(`conversation:${new_id}`)) {
        appStorage.removeItem(`conversation:${conversation.id}`);
        title_ids_storage[conversation_id] = new_id;
        conversation.id = new_id;
        add_url_to_history(`/chat/${conversation_id}`);
    }
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
        input_el.onblur = () => set_conversation_title(conversation_id, input_el.value);
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
    const conversation = await get_conversation(conversation_id);
    for (const message of conversation.items)  {
        if (Array.isArray(message.content)) {
            for (const item of message.content) {
                if ("bucket_id" in item) {
                    const delete_url = `/backend-api/v2/files/${encodeURI(item.bucket_id)}`;
                    await fetch(delete_url, {
                        method: 'DELETE'
                    });
                }
            }
        }
    }
    if (window.share_id && conversation_id == window.start_id) {
        const url = `${window.share_url}/backend-api/v2/files/${window.share_id}`;
        await fetch(url, {
            method: 'DELETE'
        });
    }
    appStorage.removeItem(`conversation:${conversation_id}`);
    const item = document.getElementById(`convo-${conversation_id}`);
    item.remove();

    if (window.conversation_id == conversation_id) {
        await new_conversation();
    }

    await load_conversations();
};

const set_conversation = async (conversation_id) => {
    if (title_ids_storage[conversation_id]) {
        conversation_id = title_ids_storage[conversation_id];
    }
    try {
        add_url_to_history(`/chat/${conversation_id}`);
    } catch (e) {
        console.error(e);
    }
    window.conversation_id = conversation_id;

    await clear_conversation();
    await load_conversation(await get_conversation(conversation_id));
    load_conversations();
    hide_sidebar(true);
};

const new_conversation = async (private = false) => {
    if (!/\/chat\/(share|\?|$)/.test(window.location.href)) {
        history.pushState({}, null, `/chat/`);
    }
    window.conversation_id = private ? null : generateUUID();
    document.title = window.title || document.title;
    document.querySelector(".chat-top-panel .convo-title").innerText = `${private ? "Private" : "New"} Conversation`;

    await clear_conversation();
    if (chatPrompt) {
        chatPrompt.value = document.getElementById("systemPrompt")?.value;
    }
    load_conversations();
    hide_sidebar(true);
    say_hello();
};

function merge_messages(message1, message2) {
    if (Array.isArray(message2)) {
        return message2;
    }
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

const load_conversation = async (conversation, scroll=true) => {
    if (!conversation) {
        return;
    }
    let messages = conversation?.items || [];
    console.debug("Conversation:", conversation.id)

    let conversation_title = conversation.new_title || conversation.title;
    title = conversation_title ? `${conversation_title} - G4F` : window.title;
    if (title) {
        document.title = title;
    }
    const chatHeader = document.querySelector(".chat-top-panel .convo-title");
    if (window.share_id && conversation.id == window.start_id) {
        chatHeader.innerHTML = '<i class="fa-solid fa-qrcode"></i> ' + escapeHtml(conversation_title);
    } else {
        chatHeader.innerText = conversation_title;
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
        buffer = filter_message_content(buffer);
        new_content = filter_message_content(item.content);
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
            if (reason != "stop") {
                actions.push("continue")
            }
        }

        add_buttons.push(`<button class="options_button">
            <div>
                <span><i class="fa-solid fa-qrcode"></i></span>
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

        countTokensEnabled = appStorage.getItem("countTokens") != "false";
        let next_usage;
        let prompt_tokens; 
        if (countTokensEnabled) {
            if (!item.continue) {
                completion_tokens = 0;
            }
            completion_tokens += item.usage?.completion_tokens ? item.usage.completion_tokens : 0;
            next_usage = messages.length > next_i ? messages[next_i].usage : null;
            prompt_tokens = next_usage?.prompt_tokens ? next_usage?.prompt_tokens : 0
        }

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
                        ${countTokensEnabled ? count_words_and_tokens(
                            item.reasoning ? item.reasoning.text + buffer : buffer,
                            next_provider?.model, completion_tokens, prompt_tokens
                        ) : ""}
                        ${add_buttons.join("")}
                    </div>
                </div>
            </div>
        `);
    });
    chatBody.innerHTML = elements.join("");

    if (suggestions) {
        const suggestions_el = document.createElement("div");
        suggestions_el.classList.add("suggestions");
        suggestions.forEach((suggestion)=> {
            const el = document.createElement("button");
            el.classList.add("suggestion");
            el.innerHTML = `<span>${escapeHtml(suggestion)}</span> <i class="fa-solid fa-turn-up"></i>`;
            el.onclick = async () => {
                await handle_ask(true, suggestion);
            }
            suggestions_el.appendChild(el);
        });
        chatBody.appendChild(suggestions_el);
        suggestions = null;
    } else if (countTokensEnabled && window.GPTTokenizer_cl100k_base) {
        const has_media = messages.filter((item)=>Array.isArray(item.content)).length > 0;
        if (!has_media) {
            const filtered = prepare_messages(messages, null, true, false);
            if (filtered.length > 0) {
                last_model = last_model?.startsWith("gpt-3") ? "gpt-3.5-turbo" : "gpt-4"
                let count_total = GPTTokenizer_cl100k_base?.encodeChat(filtered, last_model).length
                if (count_total > 0) {
                    const count_total_el = document.createElement("div");
                    count_total_el.classList.add("count_total");
                    count_total_el.innerText = `(${count_total} total tokens)`;
                    chatBody.appendChild(count_total_el);
                }
            }
        }
    }

    await register_message_buttons();
    highlight(chatBody);
    regenerate_button.classList.remove("regenerate-hidden");

    if (scroll && document.querySelector("#input-count input").checked) {
        chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

        setTimeout(() => {
            chatBody.scrollTop = chatBody.scrollHeight;
        }, 500);
        return true;
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
        let conversation = await get_conversation(conversation_id);
        return await load_conversation(conversation, scroll);
    }
}

async function get_conversation(conversation_id) {
    if (!conversation_id) {
        return privateConversation;
    }
    let conversation = await JSON.parse(
        appStorage.getItem(`conversation:${conversation_id}`)
    );
    return conversation;
}

function get_conversation_data(conversation) {
    conversation.updated = Date.now();
    return conversation;
}

async function save_conversation(conversation_id, conversation) {
    if (!conversation_id) {
        privateConversation = conversation;
        return;
    }
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
    if (!conversation_id) {
        privateConversation = {
            id: conversation_id,
            title: "",
            added: Date.now(),
            system: chatPrompt?.value,
            items: [],
        }
        return;
    }
    if (appStorage.getItem(`conversation:${conversation_id}`) == null) {
        await save_conversation(conversation_id, get_conversation_data({
            id: conversation_id,
            title: "",
            added: Date.now(),
            system: chatPrompt?.value,
            items: [],
        }));
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
        await save_conversation(window.conversation_id, get_conversation_data(conversation));
    }
}

const remove_message = async (conversation_id, index) => {
    const conversation = await get_conversation(conversation_id);
    const old_message = conversation.items[index];
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
    const data = get_conversation_data(conversation);
    await save_conversation(conversation_id, data);
    if (window.share_id && window.conversation_id == window.start_id) {
        const url = `${window.share_url}/backend-api/v2/chat/${window.share_id}`;
        await fetch(url, {
            method: 'POST',
            headers: {'content-type': 'application/json'},
            body: data,
        });
    }
    if (Array.isArray(old_message.content)) {
        for (const item of old_message.content) {
            if ("bucket_id" in item) {
                const delete_url = `/backend-api/v2/files/${encodeURI(item.bucket_id)}`;
                await fetch(delete_url, {
                    method: 'DELETE'
                });
            }
        }
    }
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
    } else if (!conversation.title && !Array.isArray(content)) {
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
    data = get_conversation_data(conversation);
    await save_conversation(conversation_id, data);
    if (window.share_id && conversation_id == window.start_id) {
        const url = `${window.share_url}/backend-api/v2/chat/${window.share_id}`;
        fetch(url, {
            method: 'POST',
            headers: {'content-type': 'application/json'},
            body: data
        });
    }
    return conversation.items.length - 1;
};

function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
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
    await clear_conversations();
    conversations.forEach((conversation) => {
        // const length = conversation.items.map((item) => (
        //     !item.content.toLowerCase().includes("hello") &&
        //     !item.content.toLowerCase().includes("hi") &&
        //     item.content
        // ) ? 1 : 0).reduce((a,b)=>a+b, 0);
        // if (!length) {
        //     appStorage.removeItem(`conversation:${conversation.id}`);
        //     return;
        // }
        const shareIcon = (conversation.id == window.start_id && window.share_id) ? '<i class="fa-solid fa-qrcode"></i>': '';
        let convo = document.createElement("div");
        convo.classList.add("convo");
        convo.id = `convo-${conversation.id}`;
        convo.innerHTML = `
            <div class="left" onclick="set_conversation('${conversation.id}')">
                <i class="fa-regular fa-comments"></i>
                <span class="datetime">${conversation.updated ? toLocaleDateString(conversation.updated) : ""}</span>
                <span class="convo-title">${shareIcon} ${escapeHtml(conversation.new_title ? conversation.new_title : conversation.title)}</span>
            </div>
            <i onclick="show_option('${conversation.id}')" class="fa-solid fa-ellipsis-vertical" id="conv-${conversation.id}"></i>
            <div id="cho-${conversation.id}" class="choise" style="display:none;">
                <i onclick="delete_conversation('${conversation.id}')" class="fa-solid fa-trash"></i>
                <i onclick="hide_option('${conversation.id}')" class="fa-regular fa-x"></i>
            </div>
        `;
        box_conversations.appendChild(convo);
    });
};

const hide_input = document.querySelector(".chat-toolbar .hide-input");
hide_input.addEventListener("click", async (e) => {
    const icon = hide_input.querySelector("i");
    const func = icon.classList.contains("fa-angles-down") ? "add" : "remove";
    const remv = icon.classList.contains("fa-angles-down") ? "remove" : "add";
    icon.classList[func]("fa-angles-up");
    icon.classList[remv]("fa-angles-down");
    document.querySelector(".chat-footer .user-input").classList[func]("hidden");
    document.querySelector(".chat-footer .buttons").classList[func]("hidden");
});

function get_message_id() {
    random_bytes = (Math.floor(Math.random() * 1338377565) + 2956589730).toString(
        2
    );
    unix = Math.floor(Date.now() / 1000).toString(2);

    return BigInt(`0b${unix}${random_bytes}`).toString();
};

async function hide_sidebar(remove_shown=false) {
    if (remove_shown) {
        sidebar.classList.remove("shown");
    }
    sidebar_buttons.forEach((el)=>el.classList.remove("rotated"))
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

sidebar_buttons.forEach((el)=>el.addEventListener("click", async () => {
    if (sidebar.classList.contains("shown") || el.classList.contains("rotated")) {
        await hide_sidebar(true);
        chat.classList.remove("hidden");
    } else {
        await show_menu();
        chat.classList.add("hidden");
    }
    window.scrollTo(0, 0);
}));

function add_url_to_history(url) {
    if (!window?.pywebview) {
        history.pushState({}, null, url);
    }
}

async function show_menu() {
    sidebar.classList.add("shown");
    sidebar_buttons.forEach((el)=>el.classList.add("rotated"))
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
        value = appStorage.getItem(element.id);
        if (value == null && element.dataset.value) {
            value = element.dataset.value;
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
                        element.placeholder = value && value.length >= 22 ? (value.substring(0, 12)+"*".repeat(12)+value.substring(value.length-12)) : "*".repeat(value ? value.length : 0);
                        element.dataset.value = value;
                    } else {
                        element.value = value == null ? element.dataset.value : value;
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

    chatBody.innerHTML += `
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
    if (!text) {
        return 0;
    }
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
    if (Array.isArray(text)) {
        return "";
    }
    text = filter_message(text);
    return `(${count_words(text)} words, ${count_chars(text)} chars, ${completion_tokens ? completion_tokens : count_tokens(model, text, prompt_tokens)} tokens)`;
}

function update_message(content_map, message_id, content = null, scroll = true) {
    content_map.update_timeouts.push(setTimeout(() => {
        if (!content) {
            if (reasoning_storage[message_id] && message_storage[message_id]) {
                content = render_reasoning(reasoning_storage[message_id], true) + markdown_render(message_storage[message_id]);
            } else if (reasoning_storage[message_id]) {
                content = render_reasoning(reasoning_storage[message_id]);
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
            content += markdown_render(`**An error occured:** ${error_storage[message_id]}`);
        }
        content_map.inner.innerHTML = content;
        if (countTokensEnabled) {
            content_map.count.innerText = count_words_and_tokens(
                (reasoning_storage[message_id] ? reasoning_storage[message_id].text : "")
                + message_storage[message_id],
                provider_storage[message_id]?.model);
        }
        highlight(content_map.inner);
        if (scroll) {
            lazy_scroll_to_bottom();
        }
        content_map.update_timeouts.forEach((timeoutId)=>clearTimeout(timeoutId));
        content_map.update_timeouts = [];
    }, 100));
};

let countFocus = userInput;
const count_input = async () => {
    if (countTokensEnabled && countFocus.value) {
        if (window.matchMedia("(pointer:coarse)")) {
            inputCount.innerText = `(${count_tokens(get_selected_model()?.value, countFocus.value)} tokens)`;
        } else {
            inputCount.innerText = count_words_and_tokens(countFocus.value, get_selected_model()?.value);
        }
    } else {
        inputCount.innerText = "";
    }
};
userInput.addEventListener("keyup", count_input);
chatPrompt.addEventListener("keyup", count_input);
chatPrompt.addEventListener("focus", function() {
    countFocus = chatPrompt;
    count_input();
});
chatPrompt.addEventListener("input", function() {
    countFocus = userInput;
    count_input();
});

window.addEventListener('load', async function() {
    if (!window.share_id) {
        return await load_conversation(JSON.parse(appStorage.getItem(`conversation:${window.conversation_id}`)));
    }
    if (!window.conversation_id) {
        window.conversation_id = window.share_id;
    }
    const response = await fetch(`${window.share_url}/backend-api/v2/chat/${window.share_id}`, {
        headers: {'accept': 'application/json', 'x-conversation-id': window.conversation_id},
    });
    if (!response.ok) {
        return await load_conversation(JSON.parse(appStorage.getItem(`conversation:${window.conversation_id}`)));
    }
    let conversation = await response.json();
    if (!appStorage.getItem(`conversation:${window.conversation_id}`) || conversation.id == window.conversation_id) {
        // Copy conversation from share
        if (conversation.id != window.conversation_id) {
            window.conversation_id = conversation.id;
            conversation.updated = Date.now();
            window.share_id = null;
        }
        await load_conversation(conversation);
        await save_conversation(conversation.id, conversation);
        await load_conversations();
        if (!window.share_id) {
            // Continue after copy conversation
            return;
        }
        let refreshOnHide = true;
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                refreshOnHide = false;
            } else {
                refreshOnHide = true;
            }
        });
        // Start chat mode (QRCode)
        var refreshIntervalId = setInterval(async () => {
            if (!window.share_id) {
                clearInterval(refreshIntervalId);
                return;
            }
            if (!refreshOnHide) {
                return;
            }
            if (window.conversation_id != window.start_id) {
                return;
            }
            const response = await fetch(`${window.share_url}/backend-api/v2/chat/${window.share_id}`, {
                headers: {
                    'accept': 'application/json',
                    'if-none-match': conversation.updated,
                    'x-conversation-id': conversation.id,
                },
            });
            if (response.status == 200) {
                const new_conversation = await response.json();
                if (conversation.id == window.conversation_id && new_conversation.updated != conversation.updated) {
                    conversation = new_conversation;
                    appStorage.setItem(
                        `conversation:${conversation.id}`,
                        JSON.stringify(conversation)
                    );
                    await load_conversations();
                    await load_conversation(conversation);
                }
            }
        }, 5000);
        return;
    }
    await safe_load_conversation(window.conversation_id, false);
});

window.addEventListener('DOMContentLoaded', async function() {
    await on_load();
    if (window.conversation_id == "{{conversation_id}}") {
        window.conversation_id = generateUUID();
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
        await load_conversations();
    } else if (/\/chat\/(share|\?|$)/.test(window.location.href)) {
        chatPrompt.value = document.getElementById("systemPrompt")?.value || "";
        chatPrompt.value = document.getElementById("systemPrompt")?.value || "";
        let chat_url = new URL(window.location.href)
        let chat_params = new URLSearchParams(chat_url.search);
        if (chat_params.get("prompt")) {
            userInput.value = chat_params.get("prompt");
            userInput.style.height = userInput.scrollHeight  + "px";
            userInput.focus();
        } else {
            await new_conversation();
        }
    } else {
        //load_conversation(window.conversation_id);
        await load_conversations();
    }
    if (window.hljs) {
        hljs.addPlugin(new HtmlRenderPlugin())
        hljs.addPlugin(new CopyButtonPlugin());
    }
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
        settings.querySelector(`.field:has(#${provider_name}-api_base)`)?.classList.remove("hidden");
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
        //settings.querySelector(`.field:has(#${provider_name}-api_key)`)?.classList.add("hidden");
    }
};

async function on_api() {
    load_version();
    let prompt_lock = false;
    userInput.addEventListener("keydown", async (evt) => {
        if (prompt_lock) return;
        // If not mobile and not shift enter
        let do_enter = userInput.value.endsWith("\n\n\n\n");
        if (do_enter || !window.matchMedia("(pointer:coarse)").matches && evt.keyCode === 13 && !evt.shiftKey) {
            evt.preventDefault();
            console.log("pressed enter");
            prompt_lock = true;
            setTimeout(()=>prompt_lock=false, 3000);
            await handle_ask(!do_enter);
        } else {
            userInput.style.height = userInput.scrollHeight  + "px";
        }
    });
    sendButton.addEventListener(`click`, async () => {
        console.log("clicked send");
        if (prompt_lock) return;
        prompt_lock = true;
        setTimeout(()=>prompt_lock=false, 3000);
        stop_recognition();
        await handle_ask();
    });
    addButton.addEventListener(`click`, async () => {
        stop_recognition();
        await handle_ask(false);
    });
    userInput.addEventListener(`click`, async () => {
        stop_recognition();
    });

    let provider_options = [];
    models = await api("models");
    models.forEach((model) => {
        let option = document.createElement("option");
        option.value = model.name;
        option.text = model.name + (model.image ? " (🖼️ Image Generation)" : "") + (model.vision ? " (👓 Image Upload)" : "") + (model.audio ? " (🎧 Audio Generation)" : "") + (model.video ? " (🎥 Video Generation)" : "");
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
            <option value="" selected="selected">Demo Mode</option>
            <option value="ARTA">ARTA Provider</option>
            <option value="DeepSeekAPI">DeepSeek Provider</option>
            <option value="Grok">Grok Provider</option>
            <option value="OpenaiChat">OpenAI Provider</option>
            <option value="PollinationsAI">Pollinations AI</option>
            <option value="G4F">G4F framework</option>
            <option value="Gemini">Gemini Provider</option>
            <option value="HuggingFace">HuggingFace</option>
            <option value="HuggingFaceMedia">HuggingFace (Image/Video Generation)</option>
            <option value="HuggingSpace">HuggingSpace</option>
            <option value="HuggingChat">HuggingChat</option>`;
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
            "HuggingFace": ["HuggingFace", "https://huggingface.co/settings/tokens", ["HuggingFaceMedia"]],
            "HuggingSpace": ["HuggingSpace", "", []],
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
                + (provider.audio ? " (Audio Generation)" : "")
                + (provider.video ? " (Video Generation)" : "")
                + (provider.nodriver ? " (Browser)" : "")
                + (provider.hf_space ? " (HuggingSpace)" : "")
                + (!provider.nodriver && provider.auth ? " (Auth)" : "");
            if (provider.parent)
                option.dataset.parent = provider.parent;
            providerSelect.appendChild(option);

            if (provider.parent) {
                if (!login_urls[provider.parent]) {
                    login_urls[provider.parent] = [provider.label, provider.login_url, [provider.name], provider.auth];
                } else {
                    login_urls[provider.parent][2].push(provider.name);
                }
            } else if (provider.login_url) {
                if (!login_urls[provider.name]) {
                    login_urls[provider.name] = [provider.label, provider.login_url, [provider.name], provider.auth];
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
                let api_key = appStorage.getItem(`${provider.name}-api_key`);
                option.innerHTML = `
                    <span class="label">Enable ${provider.label}</span>
                    <input id="Provider${provider.name}" type="checkbox" name="Provider${provider.name}" value="${provider.name}" class="provider" ${!provider.auth || api_key ? 'checked="checked"' : ''}/>
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
        <div class="collapsible-content api-key hidden"></div>
    `;
    settings.querySelector(".paper").appendChild(providersListContainer);

    for (let [name, [label, login_url, childs, auth]] of Object.entries(login_urls)) {
        if (!login_url && !is_demo) {
            continue;
        }
        let providerBox = document.createElement("div");
        providerBox.classList.add("field", "box");
        childs = childs.map((child) => `${child}-api_key`).join(" ");
        const placeholder = `placeholder="${name == "HuggingSpace" ? "zerogpu_token" : "api_key"}"`;
        providerBox.innerHTML = `
            <label for="${name}-api_key" class="label" title="">${label}:</label>
            <input type="text" id="${name}-api_key" name="${name}[api_key]" class="${childs}" ${placeholder} autocomplete="off"/>
        ` + (login_url ? `<a href="${login_url}" target="_blank" title="Login to ${label}">Get API key</a>` : "");
        if (auth) {
            providerBox.querySelector("input").addEventListener("input", (event) => {
                const input = document.getElementById(`Provider${name}`);
                input.checked = !!event.target.value;
                load_provider_option(input, name);
            });
        }
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
    const slide_systemPrompt_icon = document.querySelector(".slide-header i");
    document.querySelector(".slide-header")?.addEventListener("click", () => {
        const checked = slide_systemPrompt_icon.classList.contains("fa-angles-up");
        chatPrompt.classList[checked ? "add": "remove"]("hidden");
        slide_systemPrompt_icon.classList[checked ? "remove": "add"]("fa-angles-up");
        slide_systemPrompt_icon.classList[checked ? "add": "remove"]("fa-angles-down");
    });
    if (hide_systemPrompt.checked) {
        slide_systemPrompt_icon.click();
    }
    hide_systemPrompt.addEventListener('change', async (event) => {
        if (event.target.checked) {
            chatPrompt.classList.add("hidden");
        } else {
            chatPrompt.classList.remove("hidden");
        }
    });
    const userInputHeight = document.getElementById("message-input-height");
    if (userInputHeight) {
        if (userInputHeight.value) {
            userInput.style.maxHeight = `${userInputHeight.value}px`;
        }
        userInputHeight.addEventListener('change', async () => {
            userInput.style.maxHeight = `${userInputHeight.value}px`;
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
    if (versions["latest_version"] && versions["version"] != versions["latest_version"]) {
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

function renderMediaSelect() {
    const oldImages = mediaSelect.querySelectorAll("a:has(img)");
    oldImages.forEach((el)=>el.remove());
    Object.entries(image_storage).forEach(([object_url, file]) => {
        const link = document.createElement("a");
        link.title = file.name;
        const img = document.createElement("img");
        img.src = object_url;
        img.onclick = () => {
            img.remove();
            delete image_storage[object_url];
            if (file instanceof File) {
                URL.revokeObjectURL(object_url)
            }
        }
        img.onload = () => {
            link.title += `\n${img.naturalWidth}x${img.naturalHeight}`;
        };
        link.appendChild(img);
        mediaSelect.appendChild(link);
    });
}

imageInput.onclick = () => {
    mediaSelect.classList.toggle("hidden");
}

mediaSelect.querySelector(".close").onclick = () => {
    if (Object.values(image_storage).length) {
        Object.entries(image_storage).forEach(([object_url, file]) => {
            if (file instanceof File) {
                URL.revokeObjectURL(object_url)
            }
        });
        image_storage = {};
        renderMediaSelect();
    } else {
        mediaSelect.classList.add("hidden");
    }
}

[imageSelect, cameraInput].forEach((el) => {
    el.addEventListener('change', async () => {
        if (el.files.length) {
            Array.from(el.files).forEach((file) => {
                image_storage[URL.createObjectURL(file)] = file;
            });
            el.value = "";
            renderMediaSelect();
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

imageSelect?.addEventListener("click", (e) => {
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

function connectToSSE(url, do_refine, bucket_id) {
    const eventSource = new EventSource(url);
    eventSource.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (data.error) {
            inputCount.innerText = `Error: ${data.error.message}`;
            paperclip.classList.remove("blink");
            fileInput.value = "";
        } else if (data.action == "media") {
            inputCount.innerText = `File: ${data.filename}`;
            const url = `/files/${bucket_id}/media/${data.filename}`;
            const media = [{bucket_id: bucket_id, url: url, name: data.filename}];
            await handle_ask(false, media);
        } else if (data.action == "load") {
            inputCount.innerText = `Read data: ${formatFileSize(data.size)}`;
        } else if (data.action == "refine") {
            inputCount.innerText = `Refine data: ${formatFileSize(data.size)}`;
        } else if (data.action == "download") {
            inputCount.innerText = `Download: ${data.count} files`;
        } else if (data.action == "done") {
            if (do_refine) {
                connectToSSE(`/backend-api/v2/files/${bucket_id}?refine_chunks_with_spacy=true`, false, bucket_id);
                return;
            }
            fileInput.value = "";
            paperclip.classList.remove("blink");
            if (!data.size) {
                inputCount.innerText = "No content found";
                return
            }
            appStorage.setItem(`bucket:${bucket_id}`, data.size);
            inputCount.innerText = "Files are loaded successfully";

            const url = `/backend-api/v2/files/${bucket_id}`;
            const media = [{bucket_id: bucket_id, url: url}];
            await handle_ask(false, media);
        }
    };
    eventSource.onerror = (event) => {
        eventSource.close();
        paperclip.classList.remove("blink");
    }
}

async function upload_files(fileInput) {
    const bucket_id = generateUUID();
    paperclip.classList.add("blink");

    const formData = new FormData();
    Array.from(fileInput.files).forEach(file => {
        formData.append('files', file);
    });
    const response = await fetch("/backend-api/v2/files/" + bucket_id, {
        method: 'POST',
        body: formData
    });
    const result = await response.json()
    const count = result.files.length + result.media.length;
    inputCount.innerText = `${count} File(s) uploaded successfully`;
    if (result.files.length > 0) {
        let do_refine = document.getElementById("refine")?.checked;
        connectToSSE(`/backend-api/v2/files/${bucket_id}`, do_refine, bucket_id);
    } else {
        paperclip.classList.remove("blink");
        fileInput.value = "";
    }
    if (result.media) {
        const media = [];
        result.media.forEach((filename)=> {
            const url = `/files/${bucket_id}/media/${filename}`;
            media.push({bucket_id: bucket_id, name: filename, url: url});
        });
        await handle_ask(false, media);
    }
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
                        if (key == "options") {
                            Object.keys(data[key]).forEach(keyOption => {
                                appStorage.setItem(keyOption, data[key][keyOption]);
                                count += 1;
                            });
                        } else if (!localStorage.getItem(key)) {
                            if (key.startsWith("conversation:")) {
                                appStorage.setItem(key, JSON.stringify(data[key]));
                                count += 1;
                            } else {
                                appStorage.setItem(key, data[key]);
                            }
                        }
                    });
                    await load_conversations();
                    await load_settings_storage();
                    fileInput.value = "";
                    inputCount.innerText = `${count} Conversations/Settings were imported successfully`;
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
    if (custom_model.value) {
        return custom_model;
    } else if (modelProvider.selectedIndex >= 0) {
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
    const user = localStorage.getItem("user");
    const headers = {};
    if (user) {
        headers.x_user = user;
    }
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
        const ignored = Array.from(settings.querySelectorAll("input.provider:not(:checked)")).map((el)=>el.value);
        if (ignored) {
            headers.x_ignored = ignored.join(" ");
        }
        url = `/backend-api/v2/${ressource}/${args}`;
    } else if (ressource == "conversation") {
        let body = JSON.stringify(args);
        headers.accept = 'text/event-stream';
        if (files.length > 0) {
            const formData = new FormData();
            for (const file of files) {
                if (file instanceof File) {
                    formData.append('files', file)
                }
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
            const body = await response.text();
            const title = body.match(/<title>([^<]+?)<\/title>/)[1];
            const message = body.match(/<p>([^<]+?)<\/p>/)[1];
            error_storage[message_id] = `**${title}**\n${message}`;
            await finish_message();
            return;
        } else {
            try {
                await read_response(response, message_id, args.provider || null, scroll, finish_message);
                await finish_message();
            } catch (e) {
                console.error(e);
            }
            return;
        }
    } else if (args) {
        if (ressource == "log" ||  ressource == "usage") {
            if (ressource == "log" && !document.getElementById("report_error").checked) {
                return;
            }
            url = `https://roxky-g4f-backup.hf.space${url}`;
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
    if (!custom_model.value) {
        custom_model.classList.add("hidden");
    }
    if (provider.startsWith("Custom") || custom_model.value) {
        modelProvider.classList.add("hidden");
        modelSelect.classList.add("hidden");
        custom_model.classList.remove("hidden");
        return;
    }
    modelProvider.innerHTML = '';
    modelProvider.name = `model[${provider}]`;
    if (!provider) {
        modelProvider.classList.add("hidden");
        if (custom_model.value) {
            modelSelect.classList.add("hidden");
            custom_model.classList.remove("hidden");
        } else {
            modelSelect.classList.remove("hidden");
            custom_model.classList.add("hidden");
        }
        return;
    }
    const models = await api('models', provider);
    if (models && models.length > 0) {
        modelSelect.classList.add("hidden");
        if (!custom_model.value) {
            custom_model.classList.add("hidden");
            modelProvider.classList.remove("hidden");
        }
        let defaultIndex = 0;
        models.forEach((model, i) => {
            let option = document.createElement('option');
            option.value = model.model;
            option.dataset.label = model.model;
            option.text = model.model + (model.count > 1 ? ` (${model.count}+)` : "") + (model.image ? " (🖼️ Image Generation)" : "") + (model.vision ? " (👓 Image Upload)" : "") + (model.audio ? " (🎧 Audio Generation)" : "") + (model.video ? " (🎥 Video Generation)" : "");

            if (model.task) {
                option.text += ` (${model.task})`;
            }
            modelProvider.appendChild(option);
            if (model.default) {
                defaultIndex = i;
            }
        });
        let value = appStorage.getItem(modelProvider.name);
        if (value) {
            modelProvider.value = value;
        }
        modelProvider.selectedIndex = defaultIndex;
    } else {
        modelProvider.classList.add("hidden");
        custom_model.classList.remove("hidden")
    }
};
providerSelect.addEventListener("change", () => {
    load_provider_models()
    userInput.focus();
});
modelSelect.addEventListener("change", () => userInput.focus());
modelProvider.addEventListener("change", () =>  userInput.focus());
custom_model.addEventListener("change", () => {
    if (!custom_model.value) {
        load_provider_models();
    }
    userInput.focus();
});

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
    userInput.focus();
});

function save_storage(settings=false) {
    let filename = `${settings ? 'settings' : 'chat'} ${new Date().toLocaleString()}.json`.replaceAll(":", "-");
    let data = {"options": {"g4f": ""}};
    for (let i = 0; i < appStorage.length; i++) {
        let key = appStorage.key(i);
        let item = appStorage.getItem(key);
        if (key.startsWith("conversation:")) {
            if (!settings) {
                data[key] = JSON.parse(item);
            }
        } else if (key.startsWith("bucket:")) {
            if (!settings) {
                data[key] = item;
            }
        } else if (settings && !key.endsWith("-form") && !key.endsWith("user")) {
            data["options"][key] = item;
        } 
    }
    data = JSON.stringify(data, null, 4);
    const blob = new Blob([data], {type: 'application/json'});
    const elem = window.document.createElement('a');
    elem.href = window.URL.createObjectURL(blob);
    elem.download = filename;        
    document.body.appendChild(elem);
    elem.click();        
    document.body.removeChild(elem);
}

function import_memory() {
    if (!appStorage.getItem("mem0-api_key")) {
        return;
    }
    hide_sidebar();

    let count = 0;
    let user_id = appStorage.getItem("user") || appStorage.getItem("mem0-user_id");
    if (!user_id) {
        user_id = generateUUID();
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
        startValue = userInput.value;
        lastDebounceTranscript = "";
        userInput.readOnly = true;
        buffer = "";
    };
    recognition.onend = function() {
        if (buffer) {
            userInput.value = `${startValue ? startValue + "\n" : ""}${buffer}`;
        }
        if (microLabel.classList.contains("recognition")) {
            recognition.start();
        } else {
            userInput.readOnly = false;
            userInput.focus();
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
            userInput.value = `${startValue ? startValue + "\n" : ""}${buffer}`;
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
