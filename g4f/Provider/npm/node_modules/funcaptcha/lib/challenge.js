"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Challenge4 = exports.Challenge3 = exports.Challenge1 = exports.Challenge = void 0;
const http_1 = require("./http");
const util_1 = require("./util");
const crypt_1 = require("./crypt");
const console_1 = require("console");
class Challenge {
    constructor(data, challengeOptions) {
        this.wave = 0;
        this.data = data;
        this.userAgent = challengeOptions.userAgent;
        this.proxy = challengeOptions.proxy;
        // Preload images
        this.imgs = data.game_data.customGUI._challenge_imgs.map(async (v) => {
            let req = await (0, http_1.default)(v, {
                method: "GET",
                path: undefined,
                headers: {
                    "User-Agent": this.userAgent,
                    "Referer": this.data.tokenInfo.surl
                },
            });
            return req.body;
        });
        if (data.game_data.customGUI.encrypted_mode) {
            // Preload decryption key
            this.key = this.getKey();
        }
    }
    async getImage() {
        let img = await this.imgs[this.wave];
        try {
            JSON.parse(img.toString()); // Image is encrypted
            img = Buffer.from(await crypt_1.default.decrypt(img.toString(), await this.getKey()), "base64");
        }
        catch (err) {
            // Image is not encrypted
            // All good!
        }
        return img;
    }
    async getKey() {
        if (this.key)
            return await this.key;
        let response = await (0, http_1.default)(this.data.tokenInfo.surl, {
            method: "POST",
            path: "/fc/ekey/",
            headers: {
                "User-Agent": this.userAgent,
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": this.data.tokenInfo.surl,
            },
            body: util_1.default.constructFormData({
                session_token: this.data.session_token,
                game_token: this.data.challengeID,
            }),
        }, this.proxy);
        this.key = JSON.parse(response.body.toString()).decryption_key;
        return this.key;
    }
    get gameType() {
        return this.data.game_data.gameType;
    }
    get variant() {
        return this.data.game_data.game_variant || this.data.game_data.instruction_string;
    }
    get instruction() {
        return this.data.string_table[`${this.data.game_data.gameType}.instructions-${this.variant}`] || this.data.string_table[`${this.data.game_data.gameType}.touch_done_info${this.data.game_data.game_variant ? `_${this.data.game_data.game_variant}` : ""}`];
    }
    get waves() {
        return this.data.game_data.waves;
    }
}
exports.Challenge = Challenge;
class Challenge1 extends Challenge {
    constructor(data, challengeOptions) {
        super(data, challengeOptions);
        this.answerHistory = [];
        // But WHY?!
        let clr = data.game_data.customGUI._guiFontColr;
        this.increment = parseInt(clr ? clr.replace("#", "").substring(3) : "28", 16);
        this.increment = this.increment > 113 ? this.increment / 10 : this.increment;
    }
    round(num) {
        return (Math.round(num * 10) / 10).toFixed(2);
    }
    async answer(answer) {
        if (answer >= 0 && answer <= Math.round(360 / 51.4) - 1)
            this.answerHistory.push(this.round(answer * this.increment));
        else
            this.answerHistory.push(this.round(answer));
        let encrypted = await crypt_1.default.encrypt(this.answerHistory.toString(), this.data.session_token);
        let req = await (0, http_1.default)(this.data.tokenInfo.surl, {
            method: "POST",
            path: "/fc/ca/",
            headers: {
                "User-Agent": this.userAgent,
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": this.data.challengeURL
            },
            body: util_1.default.constructFormData({
                session_token: this.data.session_token,
                game_token: this.data.challengeID,
                guess: encrypted,
            }),
        }, this.proxy);
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }
}
exports.Challenge1 = Challenge1;
class Challenge3 extends Challenge {
    constructor(data, challengeOptions) {
        super(data, challengeOptions);
        this.answerHistory = [];
    }
    async answer(tile) {
        (0, console_1.assert)(tile >= 0 && tile <= 5, "Tile must be between 0 and 5");
        let pos = util_1.default.tileToLoc(tile);
        this.answerHistory.push(util_1.default.solveBreaker(!!this.data.game_data.customGUI.is_using_api_breaker_v2, this.data.game_data.customGUI.api_breaker, 3, pos));
        let encrypted = await crypt_1.default.encrypt(JSON.stringify(this.answerHistory), this.data.session_token);
        let requestedId = await crypt_1.default.encrypt(JSON.stringify({}), `REQUESTED${this.data.session_token}ID`);
        let { cookie: tCookie, value: tValue } = util_1.default.getTimestamp();
        let req = await (0, http_1.default)(this.data.tokenInfo.surl, {
            method: "POST",
            path: "/fc/ca/",
            headers: {
                "User-Agent": this.userAgent,
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Newrelic-Timestamp": tValue,
                "X-Requested-ID": requestedId,
                "Cookie": tCookie,
                "Referer": this.data.challengeURL
            },
            body: util_1.default.constructFormData({
                session_token: this.data.session_token,
                game_token: this.data.challengeID,
                guess: encrypted,
                analytics_tier: this.data.tokenInfo.at,
                sid: this.data.tokenInfo.r,
                bio: this.data.tokenInfo.mbio && "eyJtYmlvIjoiMTI1MCwwLDE0NywyMDQ7MTg5NCwwLDE1MSwyMDA7MTk2MCwxLDE1MiwxOTk7MjAyOSwyLDE1MiwxOTk7MjU3NSwwLDE1NSwxOTU7MjU4NSwwLDE1NiwxOTA7MjU5NSwwLDE1OCwxODU7MjYwNCwwLDE1OSwxODA7MjYxMywwLDE2MCwxNzU7MjYyMSwwLDE2MSwxNzA7MjYzMCwwLDE2MywxNjU7MjY0MCwwLDE2NCwxNjA7MjY1MCwwLDE2NSwxNTU7MjY2NCwwLDE2NiwxNTA7MjY3NywwLDE2NiwxNDQ7MjY5NCwwLDE2NywxMzk7MjcyMCwwLDE2NywxMzM7Mjc1NCwwLDE2NywxMjc7Mjc4MywwLDE2NywxMjE7MjgxMiwwLDE2NywxMTU7Mjg0MywwLDE2NywxMDk7Mjg2MywwLDE2NywxMDM7Mjg3NSwwLDE2Niw5ODsyOTA1LDAsMTY1LDkzOzMyMzIsMCwxNjUsOTk7MzI2MiwwLDE2NSwxMDU7MzI5OSwwLDE2NCwxMTA7MzM0MCwwLDE2MSwxMTU7MzM3MiwwLDE1NywxMjA7MzM5NSwwLDE1MywxMjQ7MzQwOCwwLDE0OCwxMjc7MzQyMCwwLDE0MywxMzA7MzQyOSwwLDEzOCwxMzE7MzQ0MSwwLDEzMywxMzQ7MzQ1MCwwLDEyOCwxMzU7MzQ2MSwwLDEyMywxMzg7MzQ3NiwwLDExOCwxNDA7MzQ4OSwwLDExMywxNDI7MzUwMywwLDEwOCwxNDM7MzUxOCwwLDEwMywxNDQ7MzUzNCwwLDk4LDE0NTszNTU2LDAsOTMsMTQ2OzM2MTUsMCw4OCwxNDg7MzY2MiwwLDgzLDE1MTszNjgzLDAsNzgsMTU0OzM3MDEsMCw3MywxNTc7MzcyNSwwLDY5LDE2MTszNzkzLDEsNjgsMTYyOzM4NTEsMiw2OCwxNjI7IiwidGJpbyI6IiIsImtiaW8iOiIifQ=="
            }),
        }, this.proxy);
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }
}
exports.Challenge3 = Challenge3;
class Challenge4 extends Challenge {
    constructor(data, challengeOptions) {
        super(data, challengeOptions);
        this.answerHistory = [];
    }
    async answer(index) {
        (0, console_1.assert)(index >= 0 && index <= this.data.game_data.game_difficulty - 1, "Index must be between 0 and " + (this.data.game_data.game_difficulty - 1));
        this.answerHistory.push(util_1.default.solveBreaker(!!this.data.game_data.customGUI.is_using_api_breaker_v2, this.data.game_data.customGUI.api_breaker, 4, { index }));
        let encrypted = await crypt_1.default.encrypt(JSON.stringify(this.answerHistory), this.data.session_token);
        let requestedId = await crypt_1.default.encrypt(JSON.stringify({}), `REQUESTED${this.data.session_token}ID`);
        let { cookie: tCookie, value: tValue } = util_1.default.getTimestamp();
        let req = await (0, http_1.default)(this.data.tokenInfo.surl, {
            method: "POST",
            path: "/fc/ca/",
            headers: {
                "User-Agent": this.userAgent,
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Newrelic-Timestamp": tValue,
                "X-Requested-ID": requestedId,
                "Cookie": tCookie,
                "Referer": this.data.challengeURL
            },
            body: util_1.default.constructFormData({
                session_token: this.data.session_token,
                game_token: this.data.challengeID,
                guess: encrypted,
                analytics_tier: this.data.tokenInfo.at,
                sid: this.data.tokenInfo.r,
                bio: this.data.tokenInfo.mbio && "eyJtYmlvIjoiMTI1MCwwLDE0NywyMDQ7MTg5NCwwLDE1MSwyMDA7MTk2MCwxLDE1MiwxOTk7MjAyOSwyLDE1MiwxOTk7MjU3NSwwLDE1NSwxOTU7MjU4NSwwLDE1NiwxOTA7MjU5NSwwLDE1OCwxODU7MjYwNCwwLDE1OSwxODA7MjYxMywwLDE2MCwxNzU7MjYyMSwwLDE2MSwxNzA7MjYzMCwwLDE2MywxNjU7MjY0MCwwLDE2NCwxNjA7MjY1MCwwLDE2NSwxNTU7MjY2NCwwLDE2NiwxNTA7MjY3NywwLDE2NiwxNDQ7MjY5NCwwLDE2NywxMzk7MjcyMCwwLDE2NywxMzM7Mjc1NCwwLDE2NywxMjc7Mjc4MywwLDE2NywxMjE7MjgxMiwwLDE2NywxMTU7Mjg0MywwLDE2NywxMDk7Mjg2MywwLDE2NywxMDM7Mjg3NSwwLDE2Niw5ODsyOTA1LDAsMTY1LDkzOzMyMzIsMCwxNjUsOTk7MzI2MiwwLDE2NSwxMDU7MzI5OSwwLDE2NCwxMTA7MzM0MCwwLDE2MSwxMTU7MzM3MiwwLDE1NywxMjA7MzM5NSwwLDE1MywxMjQ7MzQwOCwwLDE0OCwxMjc7MzQyMCwwLDE0MywxMzA7MzQyOSwwLDEzOCwxMzE7MzQ0MSwwLDEzMywxMzQ7MzQ1MCwwLDEyOCwxMzU7MzQ2MSwwLDEyMywxMzg7MzQ3NiwwLDExOCwxNDA7MzQ4OSwwLDExMywxNDI7MzUwMywwLDEwOCwxNDM7MzUxOCwwLDEwMywxNDQ7MzUzNCwwLDk4LDE0NTszNTU2LDAsOTMsMTQ2OzM2MTUsMCw4OCwxNDg7MzY2MiwwLDgzLDE1MTszNjgzLDAsNzgsMTU0OzM3MDEsMCw3MywxNTc7MzcyNSwwLDY5LDE2MTszNzkzLDEsNjgsMTYyOzM4NTEsMiw2OCwxNjI7IiwidGJpbyI6IiIsImtiaW8iOiIifQ=="
            }),
        }, this.proxy);
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }
    get difficulty() {
        return this.data.game_data.game_difficulty;
    }
}
exports.Challenge4 = Challenge4;
