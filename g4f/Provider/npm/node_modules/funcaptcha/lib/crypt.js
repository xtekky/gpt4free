"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const crypto_1 = require("crypto");
function encrypt(data, key) {
    let salt = "";
    let salted = "";
    let dx = Buffer.alloc(0);
    // Generate salt, as 8 random lowercase letters
    salt = String.fromCharCode(...Array(8).fill(0).map(_ => Math.floor(Math.random() * 26) + 97));
    // Our final key and iv come from the key and salt being repeatedly hashed
    // dx = md5(md5(md5(key + salt) + key + salt) + key + salt)
    // For each round of hashing, we append the result to salted, resulting in a 96 character string
    // The first 64 characters are the key, and the last 32 are the iv
    for (let x = 0; x < 3; x++) {
        dx = (0, crypto_1.createHash)("md5")
            .update(Buffer.concat([
            Buffer.from(dx),
            Buffer.from(key),
            Buffer.from(salt),
        ]))
            .digest();
        salted += dx.toString("hex");
    }
    let aes = (0, crypto_1.createCipheriv)("aes-256-cbc", Buffer.from(salted.substring(0, 64), "hex"), // Key
    Buffer.from(salted.substring(64, 64 + 32), "hex") // IV
    );
    return JSON.stringify({
        ct: aes.update(data, null, "base64") + aes.final("base64"),
        iv: salted.substring(64, 64 + 32),
        s: Buffer.from(salt).toString("hex"),
    });
}
function decrypt(rawData, key) {
    let data = JSON.parse(rawData);
    // We get our decryption key by doing the inverse of the encryption process
    let dk = Buffer.concat([Buffer.from(key), Buffer.from(data.s, "hex")]);
    let arr = [Buffer.from((0, crypto_1.createHash)("md5").update(dk).digest()).toString("hex")];
    let result = arr[0];
    for (let x = 1; x < 3; x++) {
        arr.push(Buffer.from((0, crypto_1.createHash)("md5")
            .update(Buffer.concat([Buffer.from(arr[x - 1], "hex"), dk]))
            .digest()).toString("hex"));
        result += arr[x];
    }
    let aes = (0, crypto_1.createDecipheriv)("aes-256-cbc", Buffer.from(result.substring(0, 64), "hex"), Buffer.from(data.iv, "hex"));
    return aes.update(data.ct, "base64", "utf8") + aes.final("utf8");
}
exports.default = {
    encrypt,
    decrypt,
};
