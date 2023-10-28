/// <reference types="node" />
import { RequestOptions } from "undici/types/dispatcher";
declare function req(url: string, options: RequestOptions, proxy?: string): Promise<{
    headers: import("undici/types/header").IncomingHttpHeaders;
    body: Buffer;
}>;
export default req;
