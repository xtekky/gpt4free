const query = (obj) => Object.keys(obj).map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(obj[k])).join("&");
const colorThemes = document.querySelectorAll('[name="theme"]');
const markdown = window.markdownit();
const message_box = document.getElementById(`messages`);
const message_input = document.getElementById(`message-input`);
const box_conversations = document.querySelector(`.top`);
const spinner = box_conversations.querySelector(".spinner");
const stop_generating = document.querySelector(`.stop_generating`);
const send_button = document.querySelector(`#send-button`);
let prompt_lock = false;

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

    window.scrollTo(0, 0);
    let message = message_input.value;

    if (message.length > 0) {
        message_input.value = ``;
        await ask_gpt(message);
    }
};

const remove_cancel_button = async () => {
    stop_generating.classList.add(`stop_generating-hiding`);

    setTimeout(() => {
        stop_generating.classList.remove(`stop_generating-hiding`);
        stop_generating.classList.add(`stop_generating-hidden`);
    }, 300);
};

const ask_gpt = async (message) => {
    try {
        message_input.value     = ``;
        message_input.innerHTML = ``;
        message_input.innerText = ``;

        add_conversation(window.conversation_id, message);
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
                    ${format(message)}
                </div>
            </div>
        `;

        /* .replace(/(?:\r\n|\r|\n)/g, '<br>') */

        message_box.scrollTop = message_box.scrollHeight;
        window.scrollTo(0, 0);
        await new Promise((r) => setTimeout(r, 500));
        window.scrollTo(0, 0);

        message_box.innerHTML += `
            <div class="message">
                <div class="user">
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
                "content-type": `application/json`,
                accept: `text/event-stream`,
                // v: `1.0.0`,
                // ts: Date.now().toString(),
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
                        internet_access: document.getElementById("switch").checked,
                        content_type: "text",
                        parts: [
                            {
                                content: message,
                                role: "user",
                            },
                        ],
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

            document.getElementById(`gpt_${window.token}`).innerHTML =
                markdown.render(text);
            document.querySelectorAll(`code`).forEach((el) => {
                hljs.highlightElement(el);
            });

            window.scrollTo(0, 0);
            message_box.scrollTo({ top: message_box.scrollHeight, behavior: "auto" });
        }

        if (text.includes(`G4F_ERROR`)) {
            document.getElementById(`gpt_${window.token}`).innerHTML = "An error occured, please try again, if the problem persists, please reload / refresh cache or use a differnet browser";
        }

        add_message(window.conversation_id, "user", message);
        add_message(window.conversation_id, "assistant", text);

        message_box.scrollTop = message_box.scrollHeight;
        await remove_cancel_button();
        prompt_lock = false;

        await load_conversations(20, 0);
        window.scrollTo(0, 0);
    } catch (e) {
        add_message(window.conversation_id, "user", message);

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

    conv.style.display = "none";
    yes.style.display = "block";
    not.style.display = "block";
};

const hide_option = async (conversation_id) => {
    const conv = document.getElementById(`conv-${conversation_id}`);
    const yes = document.getElementById(`yes-${conversation_id}`);
    const not = document.getElementById(`not-${conversation_id}`);

    conv.style.display = "block";
    yes.style.display = "none";
    not.style.display = "none";
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

    await make_announcement()
};

const load_conversation = async (conversation_id) => {
    let conversation = await JSON.parse(
        localStorage.getItem(`conversation:${conversation_id}`)
    );
    console.log(conversation, conversation_id);

    for (item of conversation.items) {
        message_box.innerHTML += `
            <div class="message">
                <div class="user">
                    ${item.role == "assistant" ? gpt_image : user_image}
                    ${item.role == "assistant"
                ? `<i class="fa-regular fa-phone-arrow-down-left"></i>`
                : `<i class="fa-regular fa-phone-arrow-up-right"></i>`
            }
                </div>
                <div class="content">
                    ${item.role == "assistant"
                ? markdown.render(item.content)
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
        title = content.substring(0, 17) + '..'
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
    ); // update conversation
};

const load_conversations = async (limit, offset, loader) => {
    //console.log(loader);
    //if (loader === undefined) box_conversations.appendChild(spinner);

    let conversations = [];
    for (let i = 0; i < localStorage.length; i++) {
        if (localStorage.key(i).startsWith("conversation:")) {
            let conversation = localStorage.getItem(localStorage.key(i));
            conversations.push(JSON.parse(conversation));
        }
    }

    //if (loader === undefined) spinner.parentNode.removeChild(spinner)
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

function h2a(str1) {
    var hex = str1.toString();
    var str = "";

    for (var n = 0; n < hex.length; n += 2) {
        str += String.fromCharCode(parseInt(hex.substr(n, 2), 16));
    }

    return str;
}

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

const make_announcement = async () => {
    tokens = [`Hello`, `!`, ` How`,` can`, ` I`,` assist`,` you`,` today`,`?`]

    message_box.innerHTML += `
        <div class="message">
            <div class="user">
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


(() => {
    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    
    if (check) {
        container = document.querySelector(".row")
        container.removeChild(container.querySelector('.ads'))
    }
})();

setTimeout(() => {
    ads_div = document.querySelector('.ads')

    if (ads_div != null && ads_div.getElementsByTagName("iframe").length == 0) {
        ads_div.removeChild(ads_div.querySelector('.sorry'))

        ads_div.innerHTML += `
            Please disable your adblocker to support us. <br><br>Maintaining this website costs us a lot of time and money.
        `
    }
}, 3000);

window.onload = async () => {
    load_settings_localstorage();
    setTheme();

    conversations = 0;
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

    await make_announcement()
    
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