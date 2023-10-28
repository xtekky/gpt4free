"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getToken = void 0;
const http_1 = require("./http");
const util_1 = require("./util");
async function getToken(options) {
    options = {
        surl: "https://client-api.arkoselabs.com",
        data: {},
        ...options,
    };
    if (!options.headers)
        options.headers = { "User-Agent": util_1.default.DEFAULT_USER_AGENT };
    else if (!Object.keys(options.headers).map(v => v.toLowerCase()).includes("user-agent"))
        options.headers["User-Agent"] = util_1.default.DEFAULT_USER_AGENT;
    options.headers["Accept-Language"] = "en-US,en;q=0.9";
    options.headers["Sec-Fetch-Site"] = "same-origin";
    options.headers["Accept"] = "*/*";
    options.headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8";
    options.headers["sec-fetch-mode"] = "cors";
    if (options.site) {
        options.headers["Origin"] = options.surl;
        options.headers["Referer"] = `${options.surl}/v2/${options.pkey}/1.5.5/enforcement.fbfc14b0d793c6ef8359e0e4b4a91f67.html`;
    }
    let ua = options.headers[Object.keys(options.headers).find(v => v.toLowerCase() == "user-agent")];
    let res = await (0, http_1.default)(options.surl, {
        method: "POST",
        path: "/fc/gt2/public_key/" + options.pkey,
        body: util_1.default.constructFormData({
            bda: util_1.default.getBda(ua, options),
            public_key: options.pkey,
            site: options.site ? new URL(options.site).origin : undefined,
            userbrowser: ua,
            capi_version: "1.5.5",
            capi_mode: "inline",
            style_theme: "default",
            rnd: Math.random().toString(),
            ...Object.fromEntries(Object.keys(options.data).map(v => ["data[" + v + "]", options.data[v]])),
            language: options.language || "en",
        }),
        headers: options.headers,
    }, options.proxy);
    return JSON.parse(res.body.toString());
}
exports.getToken = getToken;
