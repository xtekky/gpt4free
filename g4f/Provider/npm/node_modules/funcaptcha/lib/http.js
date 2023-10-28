"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const undici_1 = require("undici");
async function req(url, options, proxy) {
    let auth = undefined;
    if (proxy) {
        let proxyUrl = new URL(proxy);
        if (proxyUrl.username && proxyUrl.password) {
            auth = Buffer.from(proxyUrl.username + ":" + proxyUrl.password).toString("base64");
        }
    }
    let dispatcher = proxy ? new undici_1.ProxyAgent({
        uri: proxy,
        auth
    }) : undefined;
    let req = await (0, undici_1.request)(url, {
        ...options,
        dispatcher,
    });
    return {
        headers: req.headers,
        body: Buffer.from(await req.body.arrayBuffer()),
    };
}
exports.default = req;
