/// <reference types="node" />
import { TokenInfo } from "./session";
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
            };
        };
        waves: number;
        game_variant?: string;
        game_difficulty?: number;
        puzzle_name?: string;
        instruction_string?: string;
    };
    game_sid: string;
    lang: string;
    string_table: {
        [key: string]: string;
    };
    string_table_prefixes: string[];
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
export declare abstract class Challenge {
    data: ChallengeData;
    imgs: Promise<Buffer>[];
    wave: number;
    protected key: Promise<string>;
    protected userAgent: string;
    protected proxy: string;
    constructor(data: ChallengeData, challengeOptions: ChallengeOptions);
    getImage(): Promise<Buffer>;
    protected getKey(): Promise<string>;
    abstract answer(answer: number): Promise<AnswerResponse>;
    get gameType(): number;
    get variant(): string;
    get instruction(): string;
    get waves(): number;
}
export declare class Challenge1 extends Challenge {
    private answerHistory;
    increment: any;
    constructor(data: ChallengeData, challengeOptions: ChallengeOptions);
    private round;
    answer(answer: number): Promise<AnswerResponse>;
}
export declare class Challenge3 extends Challenge {
    private answerHistory;
    constructor(data: ChallengeData, challengeOptions: ChallengeOptions);
    answer(tile: number): Promise<AnswerResponse>;
}
export declare class Challenge4 extends Challenge {
    private answerHistory;
    constructor(data: ChallengeData, challengeOptions: ChallengeOptions);
    answer(index: number): Promise<AnswerResponse>;
    get difficulty(): number;
}
export {};
