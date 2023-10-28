import x64hash128 from "./murmur";
import { randomBytes } from "crypto";

const baseFingerprint = {
    DNT: "unknown", // Do not track On/Off | Previous Value: 1
    L: "en-US", // Browser language
    D: 24, // Screen color depth (in bits)
    PR: 1, // Pixel ratio
    S: [1920, 1200], // Screen resolution
    AS: [1920, 1200], // Available screen resolution
    TO: 9999, // Timezone offset
    SS: true, // Screen orientation (landscape/portrait)
    LS: true, // Local storage available
    IDB: true, // IndexedDB available
    B: false, // addBehaviour support
    ODB: true, // OpenDatabase support
    CPUC: "unknown", // CPU Class
    PK: "Win32", // Platform
    CFP: `canvas winding:yes~canvas fp:data:image/png;base64,${Buffer.from(
        Math.random().toString()
    ).toString("base64")}`, // Canvas fingerprint (if canvas is supported)
    FR: false, // Fake screen resolution?
    FOS: false, // Fake OS?
    FB: false, // Fake Browser?
    JSF: [
        "Andale Mono",
        "Arial",
        "Arial Black",
        "Arial Hebrew",
        "Arial MT",
        "Arial Narrow",
        "Arial Rounded MT Bold",
        "Arial Unicode MS",
        "Bitstream Vera Sans Mono",
        "Book Antiqua",
        "Bookman Old Style",
        "Calibri",
        "Cambria",
        "Cambria Math",
        "Century",
        "Century Gothic",
        "Century Schoolbook",
        "Comic Sans",
        "Comic Sans MS",
        "Consolas",
        "Courier",
        "Courier New",
        "Garamond",
        "Geneva",
        "Georgia",
        "Helvetica",
        "Helvetica Neue",
        "Impact",
        "Lucida Bright",
        "Lucida Calligraphy",
        "Lucida Console",
        "Lucida Fax",
        "LUCIDA GRANDE",
        "Lucida Handwriting",
        "Lucida Sans",
        "Lucida Sans Typewriter",
        "Lucida Sans Unicode",
        "Microsoft Sans Serif",
        "Monaco",
        "Monotype Corsiva",
        "MS Gothic",
        "MS Outlook",
        "MS PGothic",
        "MS Reference Sans Serif",
        "MS Sans Serif",
        "MS Serif",
        "MYRIAD",
        "MYRIAD PRO",
        "Palatino",
        "Palatino Linotype",
        "Segoe Print",
        "Segoe Script",
        "Segoe UI",
        "Segoe UI Light",
        "Segoe UI Semibold",
        "Segoe UI Symbol",
        "Tahoma",
        "Times",
        "Times New Roman",
        "Times New Roman PS",
        "Trebuchet MS",
        "Verdana",
        "Wingdings",
        "Wingdings 2",
        "Wingdings 3",
    ], // Available fonts
    P: [
        "Chrome PDF Plugin::Portable Document Format::application/x-google-chrome-pdf~pdf",
        "Chrome PDF Viewer::::application/pdf~pdf",
        "Native Client::::application/x-nacl~,application/x-pnacl~",
    ], // Plugins
    T: [0, false, false], // Touch screen (maxTouchPoints, TouchEvent event listener support, ontouchstart support)
    H: 24, // Cpu threads
    SWF: false, // Flash support
};

const languages = [
    "af", "af-ZA", "ar", "ar-AE", "ar-BH", "ar-DZ", "ar-EG", "ar-IQ", "ar-JO", "ar-KW", "ar-LB", "ar-LY", "ar-MA", "ar-OM", "ar-QA", "ar-SA",
    "ar-SY", "ar-TN", "ar-YE", "az", "az-AZ", "az-AZ", "be", "be-BY", "bg", "bg-BG", "bs-BA", "ca", "ca-ES", "cs", "cs-CZ", "cy",
    "cy-GB", "da", "da-DK", "de", "de-AT", "de-CH", "de-DE", "de-LI", "de-LU", "dv", "dv-MV", "el", "el-GR", "en", "en-AU", "en-BZ",
    "en-CA", "en-CB", "en-GB", "en-IE", "en-JM", "en-NZ", "en-PH", "en-TT", "en-US", "en-ZA", "en-ZW", "eo", "es", "es-AR", "es-BO", "es-CL",
    "es-CO", "es-CR", "es-DO", "es-EC", "es-ES", "es-ES", "es-GT", "es-HN", "es-MX", "es-NI", "es-PA", "es-PE", "es-PR", "es-PY", "es-SV", "es-UY",
    "es-VE", "et", "et-EE", "eu", "eu-ES", "fa", "fa-IR", "fi", "fi-FI", "fo", "fo-FO", "fr", "fr-BE", "fr-CA", "fr-CH", "fr-FR",
    "fr-LU", "fr-MC", "gl", "gl-ES", "gu", "gu-IN", "he", "he-IL", "hi", "hi-IN", "hr", "hr-BA", "hr-HR", "hu", "hu-HU", "hy",
    "hy-AM", "id", "id-ID", "is", "is-IS", "it", "it-CH", "it-IT", "ja", "ja-JP", "ka", "ka-GE", "kk", "kk-KZ", "kn", "kn-IN",
    "ko", "ko-KR", "kok", "kok-IN", "ky", "ky-KG", "lt", "lt-LT", "lv", "lv-LV", "mi", "mi-NZ", "mk", "mk-MK", "mn", "mn-MN",
    "mr", "mr-IN", "ms", "ms-BN", "ms-MY", "mt", "mt-MT", "nb", "nb-NO", "nl", "nl-BE", "nl-NL", "nn-NO", "ns", "ns-ZA", "pa",
    "pa-IN", "pl", "pl-PL", "ps", "ps-AR", "pt", "pt-BR", "pt-PT", "qu", "qu-BO", "qu-EC", "qu-PE", "ro", "ro-RO", "ru", "ru-RU",
    "sa", "sa-IN", "se", "se-FI", "se-FI", "se-FI", "se-NO", "se-NO", "se-NO", "se-SE", "se-SE", "se-SE", "sk", "sk-SK", "sl", "sl-SI",
    "sq", "sq-AL", "sr-BA", "sr-BA", "sr-SP", "sr-SP", "sv", "sv-FI", "sv-SE", "sw", "sw-KE", "syr", "syr-SY", "ta", "ta-IN", "te",
    "te-IN", "th", "th-TH", "tl", "tl-PH", "tn", "tn-ZA", "tr", "tr-TR", "tt", "tt-RU", "ts", "uk", "uk-UA", "ur", "ur-PK",
    "uz", "uz-UZ", "uz-UZ", "vi", "vi-VN", "xh", "xh-ZA", "zh", "zh-CN", "zh-HK", "zh-MO", "zh-SG", "zh-TW", "zu", "zu-ZA"
];

let screenRes = [
    [1920, 1080],
    [1920, 1200],
    [2048, 1080],
    [2560, 1440],
    [1366, 768],
    [1440, 900],
    [1536, 864],
    [1680, 1050],
    [1280, 1024],
    [1280, 800],
    [1280, 720],
    [1600, 1200],
    [1600, 900],
];
function randomScreenRes() {
    return screenRes[Math.floor(Math.random() * screenRes.length)];
}

// Get fingerprint
function getFingerprint() {
    let fingerprint = { ...baseFingerprint }; // Create a copy of the base fingerprint

    // Randomization time!
    fingerprint["DNT"] = "unknown";
    fingerprint["L"] = languages[Math.floor(Math.random() * languages.length)];
    fingerprint["D"] = [8, 24][
        Math.floor(Math.random() * 2)
    ];
    fingerprint["PR"] = Math.round(Math.random() * 100) / 100 * 2 + 0.5;
    fingerprint["S"] = randomScreenRes();
    fingerprint["AS"] = fingerprint.S;
    fingerprint["TO"] = (Math.floor(Math.random() * 24) - 12) * 60;
    fingerprint["SS"] = Math.random() > 0.5;
    fingerprint["LS"] = Math.random() > 0.5;
    fingerprint["IDB"] = Math.random() > 0.5;
    fingerprint["B"] = Math.random() > 0.5;
    fingerprint["ODB"] = Math.random() > 0.5;
    fingerprint["CPUC"] = "unknown";
    fingerprint["PK"] = "Win32"
    fingerprint["CFP"] = "canvas winding:yes~canvas fp:data:image/png;base64," + randomBytes(128).toString("base64");
    fingerprint["FR"] = false; // Fake Resolution
    fingerprint["FOS"] = false; // Fake Operating System
    fingerprint["FB"] = false; // Fake Browser
    fingerprint["JSF"] = fingerprint["JSF"].filter(() => Math.random() > 0.5);
    fingerprint["P"] = fingerprint["P"].filter(() => Math.random() > 0.5);
    fingerprint["T"] = [
        Math.floor(Math.random() * 8),
        Math.random() > 0.5,
        Math.random() > 0.5,
    ];
    fingerprint["H"] = 2 ** Math.floor(Math.random() * 6);
    fingerprint["SWF"] = fingerprint["SWF"]; // RIP Flash

    return fingerprint;
}

function prepareF(fingerprint) {
    let f = [];
    let keys = Object.keys(fingerprint);
    for (let i = 0; i < keys.length; i++) {
        if (fingerprint[keys[i]].join) f.push(fingerprint[keys[i]].join(";"));
        else f.push(fingerprint[keys[i]]);
    }
    return f.join("~~~");
}

function prepareFe(fingerprint) {
    let fe = [];
    let keys = Object.keys(fingerprint);
    for (let i = 0; i < keys.length; i++) {
        switch (keys[i]) {
            case "CFP":
                fe.push(`${keys[i]}:${cfpHash(fingerprint[keys[i]])}`);
                break;
            case "P":
                fe.push(
                    `${keys[i]}:${fingerprint[keys[i]].map(
                        (v) => v.split("::")[0]
                    )}`
                );
                break;
            default:
                fe.push(`${keys[i]}:${fingerprint[keys[i]]}`);
                break;
        }
    }
    return fe;
}

function cfpHash(H8W) {
    var l8W, U8W;
    if (!H8W) return "";
    if (Array.prototype.reduce)
        return H8W.split("").reduce(function (p8W, z8W) {
            p8W = (p8W << 5) - p8W + z8W.charCodeAt(0);
            return p8W & p8W;
        }, 0);
    l8W = 0;
    if (H8W.length === 0) return l8W;
    for (var k8W = 0; k8W < H8W.length; k8W++) {
        U8W = H8W.charCodeAt(k8W);
        l8W = (l8W << 5) - l8W + U8W;
        l8W = l8W & l8W;
    }
    return l8W;
}

let baseEnhancedFingerprint = {
    "webgl_extensions": "ANGLE_instanced_arrays;EXT_blend_minmax;EXT_color_buffer_half_float;EXT_disjoint_timer_query;EXT_float_blend;EXT_frag_depth;EXT_shader_texture_lod;EXT_texture_compression_bptc;EXT_texture_compression_rgtc;EXT_texture_filter_anisotropic;EXT_sRGB;KHR_parallel_shader_compile;OES_element_index_uint;OES_fbo_render_mipmap;OES_standard_derivatives;OES_texture_float;OES_texture_float_linear;OES_texture_half_float;OES_texture_half_float_linear;OES_vertex_array_object;WEBGL_color_buffer_float;WEBGL_compressed_texture_s3tc;WEBGL_compressed_texture_s3tc_srgb;WEBGL_debug_renderer_info;WEBGL_debug_shaders;WEBGL_depth_texture;WEBGL_draw_buffers;WEBGL_lose_context;WEBGL_multi_draw",
    "webgl_extensions_hash": "58a5a04a5bef1a78fa88d5c5098bd237",
    "webgl_renderer": "WebKit WebGL",
    "webgl_vendor": "WebKit",
    "webgl_version": "WebGL 1.0 (OpenGL ES 2.0 Chromium)",
    "webgl_shading_language_version": "WebGL GLSL ES 1.0 (OpenGL ES GLSL ES 1.0 Chromium)",
    "webgl_aliased_line_width_range": "[1, 1]",
    "webgl_aliased_point_size_range": "[1, 1023]",
    "webgl_antialiasing": "yes",
    "webgl_bits": "8,8,24,8,8,0",
    "webgl_max_params": "16,64,16384,4096,8192,32,8192,31,16,32,4096",
    "webgl_max_viewport_dims": "[8192, 8192]",
    "webgl_unmasked_vendor": "Google Inc. (Google)",
    "webgl_unmasked_renderer": "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
    "webgl_vsf_params": "23,127,127,23,127,127,23,127,127",
    "webgl_vsi_params": "0,31,30,0,31,30,0,31,30",
    "webgl_fsf_params": "23,127,127,23,127,127,23,127,127",
    "webgl_fsi_params": "0,31,30,0,31,30,0,31,30",
    "webgl_hash_webgl": null,
    "user_agent_data_brands": "Chromium,Google Chrome,Not=A?Brand",
    "user_agent_data_mobile": null,
    "navigator_connection_downlink": null,
    "navigator_connection_downlink_max": null,
    "network_info_rtt": null,
    "network_info_save_data": false,
    "network_info_rtt_type": null,
    "screen_pixel_depth": 24,
    "navigator_device_memory": 0.5,
    "navigator_languages": "en-US,fr-BE,fr,en-BE,en",
    "window_inner_width": 0,
    "window_inner_height": 0,
    "window_outer_width": 2195,
    "window_outer_height": 1195,
    "browser_detection_firefox": false,
    "browser_detection_brave": false,
    "audio_codecs": "{\"ogg\":\"probably\",\"mp3\":\"probably\",\"wav\":\"probably\",\"m4a\":\"maybe\",\"aac\":\"probably\"}",
    "video_codecs": "{\"ogg\":\"probably\",\"h264\":\"probably\",\"webm\":\"probably\",\"mpeg4v\":\"\",\"mpeg4a\":\"\",\"theora\":\"\"}",
    "media_query_dark_mode": true,
    "headless_browser_phantom": false,
    "headless_browser_selenium": false,
    "headless_browser_nightmare_js": false,
    "document__referrer": "https://www.roblox.com/",
    "window__ancestor_origins": [
        "https://www.roblox.com",
    ],
    "window__tree_index": [
        0
    ],
    "window__tree_structure": "[[]]",
    "window__location_href": "https://roblox-api.arkoselabs.com/v2/1.5.5/enforcement.fbfc14b0d793c6ef8359e0e4b4a91f67.html#476068BF-9607-4799-B53D-966BE98E2B81",
    "client_config__sitedata_location_href": "https://www.roblox.com/arkose/iframe",
    "client_config__surl": "https://roblox-api.arkoselabs.com",
    "client_config__language": null,
    "navigator_battery_charging": true,
    "audio_fingerprint": "124.04347527516074"
}
function getEnhancedFingerprint(fp: typeof baseFingerprint, ua: string, opts: any) {
    let fingerprint = { ...baseEnhancedFingerprint };

    fingerprint.webgl_extensions = fingerprint.webgl_extensions.split(";").filter(_ => Math.random() > 0.5).join(";");
    fingerprint.webgl_extensions_hash = x64hash128(fingerprint.webgl_extensions, 0);
    fingerprint.screen_pixel_depth = fp.D;
    fingerprint.navigator_languages = fp.L;
    fingerprint.window_outer_height = fp.S[0];
    fingerprint.window_outer_width = fp.S[1];
    fingerprint.window_inner_height = fp.S[0];
    fingerprint.window_inner_width = fp.S[1];
    fingerprint.screen_pixel_depth = fp.D;
    fingerprint.browser_detection_firefox = !!ua.match(/Firefox\/\d+/)
    fingerprint.browser_detection_brave = !!ua.match(/Brave\/\d+/)
    fingerprint.media_query_dark_mode = Math.random() > 0.9;
    fingerprint.webgl_hash_webgl = x64hash128(Object.entries(fingerprint).filter(([k, v]) => k.startsWith("webgl_") && k != "webgl_hash_webgl").map(([k, v]) => v).join(","), 0);

    fingerprint.client_config__language = opts.language || null;
    fingerprint.window__location_href = `${opts.surl}/v2/1.5.5/enforcement.fbfc14b0d793c6ef8359e0e4b4a91f67.html#${opts.pkey}`
    if (opts.site) {
        fingerprint.document__referrer = opts.site;
        fingerprint.window__ancestor_origins = [opts.site];
        fingerprint.client_config__sitedata_location_href = opts.site;
    }

    fingerprint.client_config__surl = opts.surl || "https://client-api.arkoselabs.com";
    fingerprint.audio_fingerprint = (124.04347527516074 + Math.random() * 0.001 - 0.0005).toString();
    
    return Object.entries(fingerprint).map(([k, v]) => ({ key: k, value: v }));
}

export default {
    getFingerprint,
    prepareF,
    prepareFe,
    getEnhancedFingerprint,
};
