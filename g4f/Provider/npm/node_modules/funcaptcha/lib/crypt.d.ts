declare function encrypt(data: string, key: string): string;
declare function decrypt(rawData: string, key: string): string;
declare const _default: {
    encrypt: typeof encrypt;
    decrypt: typeof decrypt;
};
export default _default;
