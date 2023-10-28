import request from "./http";
import { TokenInfo } from "./session";
import util from "./util";
import crypt from "./crypt";
import { assert } from "console";

interface ChallengeOptions {
    userAgent?: string;
    proxy?: string;
}

interface ChallengeData {
    token: string;
    tokenInfo: TokenInfo;
    session_token: string;
    challengeID: string;
    challengeURL: string;
    game_data: {
        gameType: number;
        customGUI: {
            is_using_api_breaker_v2: boolean;
            _guiFontColr: string;
            _challenge_imgs: string[];
            api_breaker: string;
            encrypted_mode: number;
            example_images: {
                correct: string;
                incorrect: string;
            }
        };
        waves: number;
        game_variant?: string; // For gametype 3
        game_difficulty?: number;
        puzzle_name?: string; // For gametype 4
        instruction_string?: string; // For gametype 4
    };
    game_sid: string;
    lang: string;
    string_table: {
        [key: string]: string;
    },
    string_table_prefixes: string[]
}

interface AnswerResponse {
    response: "not answered" | "answered";
    solved?: boolean;
    incorrect_guess?: number;
    score?: number;
    decryption_key?: string;
    time_end?: number;
    time_end_seconds?: number;
}

export abstract class Challenge {
    public data: ChallengeData;
    public imgs: Promise<Buffer>[];
    public wave: number = 0;
    protected key: Promise<string>;
    protected userAgent: string;
    protected proxy: string;

    constructor(data: ChallengeData, challengeOptions: ChallengeOptions) {
        this.data = data;
        this.userAgent = challengeOptions.userAgent;
        this.proxy = challengeOptions.proxy;

        // Preload images
        this.imgs = data.game_data.customGUI._challenge_imgs.map(async (v) => {
            let req = await request(v, {
                method: "GET",
                path: undefined,
                headers: {
                    "User-Agent": this.userAgent,
                    "Referer": this.data.tokenInfo.surl
                },
            });
            return req.body;
        });

        if(data.game_data.customGUI.encrypted_mode) {
            // Preload decryption key
            this.key = this.getKey();
        }
    }

    async getImage(): Promise<Buffer> {
        let img = await this.imgs[this.wave];
        try {
            JSON.parse(img.toString()); // Image is encrypted
            img = Buffer.from(
                await crypt.decrypt(img.toString(), await this.getKey()),
                "base64"
            );
        } catch (err) {
            // Image is not encrypted
            // All good!
        }
        return img;
    }

    protected async getKey() {
        if (this.key) return await this.key;
        let response = await request(
            this.data.tokenInfo.surl,
            {
                method: "POST",
                path: "/fc/ekey/",
                headers: {
                    "User-Agent": this.userAgent,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": this.data.tokenInfo.surl,
                },
                body: util.constructFormData({
                    session_token: this.data.session_token,
                    game_token: this.data.challengeID,
                }),
            },
            this.proxy
        );
        this.key = JSON.parse(response.body.toString()).decryption_key;
        return this.key;
    }

    abstract answer(answer: number): Promise<AnswerResponse>;

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

export class Challenge1 extends Challenge {
    private answerHistory = [];
    public increment;

    constructor(data: ChallengeData, challengeOptions: ChallengeOptions) {
        super(data, challengeOptions);
        
        // But WHY?!
        let clr = data.game_data.customGUI._guiFontColr
        this.increment = parseInt(clr ? clr.replace("#", "").substring(3) : "28", 16)
        this.increment = this.increment > 113 ? this.increment / 10 : this.increment
    }

    private round(num: number): string {
        return (Math.round(num * 10) / 10).toFixed(2);
    }

    async answer(answer: number): Promise<AnswerResponse> {
        if(answer >= 0 && answer <= Math.round(360 / 51.4) - 1)
            this.answerHistory.push(this.round(answer * this.increment));
        else
            this.answerHistory.push(this.round(answer))

        let encrypted = await crypt.encrypt(
            this.answerHistory.toString(),
            this.data.session_token
        );
        let req = await request(
            this.data.tokenInfo.surl,
            {
                method: "POST",
                path: "/fc/ca/",
                headers: {
                    "User-Agent": this.userAgent,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": this.data.challengeURL
                },
                body: util.constructFormData({
                    session_token: this.data.session_token,
                    game_token: this.data.challengeID,
                    guess: encrypted,
                }),
            },
            this.proxy
        );
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }
}

export class Challenge3 extends Challenge {
    private answerHistory = [];

    constructor(data: ChallengeData, challengeOptions: ChallengeOptions) {
        super(data, challengeOptions);
    }

    async answer(tile: number): Promise<AnswerResponse> {
        assert(tile >= 0 && tile <= 5, "Tile must be between 0 and 5");
        
        let pos = util.tileToLoc(tile);
        this.answerHistory.push(util.solveBreaker(!!this.data.game_data.customGUI.is_using_api_breaker_v2, this.data.game_data.customGUI.api_breaker, 3, pos))
        
        let encrypted = await crypt.encrypt(
            JSON.stringify(this.answerHistory),
            this.data.session_token
        );
        let requestedId = await crypt.encrypt(JSON.stringify({}), `REQUESTED${this.data.session_token}ID`);
        let { cookie: tCookie, value: tValue } = util.getTimestamp();
        let req = await request(
            this.data.tokenInfo.surl,
            {
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
                body: util.constructFormData({
                    session_token: this.data.session_token,
                    game_token: this.data.challengeID,
                    guess: encrypted,
                    analytics_tier: this.data.tokenInfo.at,
                    sid: this.data.tokenInfo.r,
                    bio: this.data.tokenInfo.mbio && "eyJtYmlvIjoiMTI1MCwwLDE0NywyMDQ7MTg5NCwwLDE1MSwyMDA7MTk2MCwxLDE1MiwxOTk7MjAyOSwyLDE1MiwxOTk7MjU3NSwwLDE1NSwxOTU7MjU4NSwwLDE1NiwxOTA7MjU5NSwwLDE1OCwxODU7MjYwNCwwLDE1OSwxODA7MjYxMywwLDE2MCwxNzU7MjYyMSwwLDE2MSwxNzA7MjYzMCwwLDE2MywxNjU7MjY0MCwwLDE2NCwxNjA7MjY1MCwwLDE2NSwxNTU7MjY2NCwwLDE2NiwxNTA7MjY3NywwLDE2NiwxNDQ7MjY5NCwwLDE2NywxMzk7MjcyMCwwLDE2NywxMzM7Mjc1NCwwLDE2NywxMjc7Mjc4MywwLDE2NywxMjE7MjgxMiwwLDE2NywxMTU7Mjg0MywwLDE2NywxMDk7Mjg2MywwLDE2NywxMDM7Mjg3NSwwLDE2Niw5ODsyOTA1LDAsMTY1LDkzOzMyMzIsMCwxNjUsOTk7MzI2MiwwLDE2NSwxMDU7MzI5OSwwLDE2NCwxMTA7MzM0MCwwLDE2MSwxMTU7MzM3MiwwLDE1NywxMjA7MzM5NSwwLDE1MywxMjQ7MzQwOCwwLDE0OCwxMjc7MzQyMCwwLDE0MywxMzA7MzQyOSwwLDEzOCwxMzE7MzQ0MSwwLDEzMywxMzQ7MzQ1MCwwLDEyOCwxMzU7MzQ2MSwwLDEyMywxMzg7MzQ3NiwwLDExOCwxNDA7MzQ4OSwwLDExMywxNDI7MzUwMywwLDEwOCwxNDM7MzUxOCwwLDEwMywxNDQ7MzUzNCwwLDk4LDE0NTszNTU2LDAsOTMsMTQ2OzM2MTUsMCw4OCwxNDg7MzY2MiwwLDgzLDE1MTszNjgzLDAsNzgsMTU0OzM3MDEsMCw3MywxNTc7MzcyNSwwLDY5LDE2MTszNzkzLDEsNjgsMTYyOzM4NTEsMiw2OCwxNjI7IiwidGJpbyI6IiIsImtiaW8iOiIifQ=="
                }),
            },
            this.proxy
        );
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }
}

export class Challenge4 extends Challenge {
    private answerHistory = [];

    constructor(data: ChallengeData, challengeOptions: ChallengeOptions) {
        super(data, challengeOptions);
    }

    async answer(index: number): Promise<AnswerResponse> {
        assert(index >= 0 && index <= this.data.game_data.game_difficulty - 1, "Index must be between 0 and " + (this.data.game_data.game_difficulty - 1));
        this.answerHistory.push(util.solveBreaker(!!this.data.game_data.customGUI.is_using_api_breaker_v2, this.data.game_data.customGUI.api_breaker, 4, { index }))
        
        let encrypted = await crypt.encrypt(
            JSON.stringify(this.answerHistory),
            this.data.session_token
        );
        let requestedId = await crypt.encrypt(JSON.stringify({}), `REQUESTED${this.data.session_token}ID`);
        let { cookie: tCookie, value: tValue } = util.getTimestamp();
        let req = await request(
            this.data.tokenInfo.surl,
            {
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
                body: util.constructFormData({
                    session_token: this.data.session_token,
                    game_token: this.data.challengeID,
                    guess: encrypted,
                    analytics_tier: this.data.tokenInfo.at,
                    sid: this.data.tokenInfo.r,
                    bio: this.data.tokenInfo.mbio && "eyJtYmlvIjoiMTI1MCwwLDE0NywyMDQ7MTg5NCwwLDE1MSwyMDA7MTk2MCwxLDE1MiwxOTk7MjAyOSwyLDE1MiwxOTk7MjU3NSwwLDE1NSwxOTU7MjU4NSwwLDE1NiwxOTA7MjU5NSwwLDE1OCwxODU7MjYwNCwwLDE1OSwxODA7MjYxMywwLDE2MCwxNzU7MjYyMSwwLDE2MSwxNzA7MjYzMCwwLDE2MywxNjU7MjY0MCwwLDE2NCwxNjA7MjY1MCwwLDE2NSwxNTU7MjY2NCwwLDE2NiwxNTA7MjY3NywwLDE2NiwxNDQ7MjY5NCwwLDE2NywxMzk7MjcyMCwwLDE2NywxMzM7Mjc1NCwwLDE2NywxMjc7Mjc4MywwLDE2NywxMjE7MjgxMiwwLDE2NywxMTU7Mjg0MywwLDE2NywxMDk7Mjg2MywwLDE2NywxMDM7Mjg3NSwwLDE2Niw5ODsyOTA1LDAsMTY1LDkzOzMyMzIsMCwxNjUsOTk7MzI2MiwwLDE2NSwxMDU7MzI5OSwwLDE2NCwxMTA7MzM0MCwwLDE2MSwxMTU7MzM3MiwwLDE1NywxMjA7MzM5NSwwLDE1MywxMjQ7MzQwOCwwLDE0OCwxMjc7MzQyMCwwLDE0MywxMzA7MzQyOSwwLDEzOCwxMzE7MzQ0MSwwLDEzMywxMzQ7MzQ1MCwwLDEyOCwxMzU7MzQ2MSwwLDEyMywxMzg7MzQ3NiwwLDExOCwxNDA7MzQ4OSwwLDExMywxNDI7MzUwMywwLDEwOCwxNDM7MzUxOCwwLDEwMywxNDQ7MzUzNCwwLDk4LDE0NTszNTU2LDAsOTMsMTQ2OzM2MTUsMCw4OCwxNDg7MzY2MiwwLDgzLDE1MTszNjgzLDAsNzgsMTU0OzM3MDEsMCw3MywxNTc7MzcyNSwwLDY5LDE2MTszNzkzLDEsNjgsMTYyOzM4NTEsMiw2OCwxNjI7IiwidGJpbyI6IiIsImtiaW8iOiIifQ=="
                }),
            },
            this.proxy
        );
        let reqData = JSON.parse(req.body.toString());
        this.key = reqData.decryption_key || "";
        this.wave++;
        return reqData;
    }

    get difficulty(): number {
        return this.data.game_data.game_difficulty;
    }
}