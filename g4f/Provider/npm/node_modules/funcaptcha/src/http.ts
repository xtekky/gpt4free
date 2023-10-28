import { request, ProxyAgent } from "undici";
// @ts-ignore
import { RequestOptions } from "undici/types/dispatcher";

async function req(url: string, options: RequestOptions, proxy?: string) {
    let auth = undefined;
    if (proxy) {
        let proxyUrl = new URL(proxy);
        if(proxyUrl.username && proxyUrl.password) {
            auth = Buffer.from(proxyUrl.username + ":" + proxyUrl.password).toString("base64")
        }
    }
    let dispatcher = proxy ? new ProxyAgent({
        uri: proxy,
        auth
    }) : undefined;

    let req = await request(url, {
        ...options,
        dispatcher,
    });
    return {
        headers: req.headers,
        body: Buffer.from(await req.body.arrayBuffer()),
    };
}

export default req;
