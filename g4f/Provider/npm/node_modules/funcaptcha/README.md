# funcaptcha
A library used to interact with funcaptchas.
## Installation
This package is available on npm.  
Simply run: `npm install funcaptcha`
## Usage And Documentation
Require the library like any other
```js
const fun = require("funcaptcha")
```

You must first fetch a token using getToken
```js
const token = await fun.getToken({
    pkey: "476068BF-9607-4799-B53D-966BE98E2B81", // The public key
    surl: "https://roblox-api.arkoselabs.com", // OPTIONAL: Some websites can have a custom service URL
    data: { // OPTIONAL
        blob: "blob" // Some websites can have custom data passed: here it is data[blob]
    },
    headers: { // OPTIONAL
        // You can pass custom headers if you have to, but keep
        // in mind to pass a user agent when doing that
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    },
    site: "https://www.roblox.com/login", // The site which contains the funcaptcha
    proxy: "http://127.0.0.1:8888" // OPTIONAL: A proxy to fetch the token, usually not required
    // NOTE: The proxy will only be used for fetching the token, and not future requests such as getting images and answering captchas
})
```

You can then create a new session
```js
// Token, in this case, may either be a string (if you already know it) or an object you received from getToken (it will strip the token out of the object)
const session = new fun.Session(token, {
    proxy: "http://127.0.0.1:8888", // OPTIONAL: A proxy used to get images and answer captchas, usually not required
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36" // OPTIONAL: Custom user agent for all future requests
})

// If you would like to let a user solve the captcha in their browser
// NOTE: Embed URLs will not work unless put in an iframe.
console.log(session.getEmbedUrl())

// Suppressed captchas are instantly valid an do not require for you to load a challenge (it will error)
// These can occur when using a trusted IP and good fingerprint
// You can check if a captcha is suppressed by doing the following:
console.log(session.tokenInfo.sup == "1")
```

One session can get you 10 funcaptcha challenges, you will have to get another session after that.
```js
let challenge = await session.getChallenge()
// Please view https://pastebin.com/raw/Gi6yKwyD to see all the data you can find 
console.log(challenge.gameType) // Gets the game type (ball, tiles, matchkey, etc...)
console.log(challenge.variant) // The game variant, eg: apple, rotated, maze, dice_pair, dart, card, 3d_rollball_animals, etc...
console.log(challenge.instruction) // Self explanatory
console.log(challenge.waves) // Wave count
console.log(challenge.wave) // Current wave number

// You can then use these functions
await challenge.getImage()

// For game type 1, where you have to rotate a circle to put the image in the correct orientation
// In this game type, the angle increment can vary. It can be found with challenge.increment
await challenge.answer(3) // Usually 0-6, but can be 0-5 or 0-6 depending on challenge.increment (clockwise)
await challenge.answer(51.4) // You can input the raw angle as well (clockwise, negative for counter clockwise)

// For game type 3, where you have to pick one of 6 tiles
await challenge.answer(2) // 0-5, please see https://github.com/noahcoolboy/roblox-funcaptcha/raw/master/img.gif

// For game type 4, where you pick an image from a selection of images which matches the prompt compared to the image on the left
// The answer should be between 0 and challenge.difficulty
await challenge.answer(2) // Pick the third image
```

## Full Example
```js
const fs = require("fs")
const fun = require("funcaptcha")
const readline = require("readline")
let rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
})

function ask(question) {
    return new Promise((resolve, reject) => {
        rl.question(question, (answer) => {
            resolve(answer)
        })
    })
}

fun.getToken({
    pkey: "69A21A01-CC7B-B9C6-0F9A-E7FA06677FFC",
}).then(async token => { 
    let session = new fun.Session(token)
    let challenge = await session.getChallenge()
    console.log(challenge.data.game_data.game_variant)
    console.log(challenge.data.game_data.customGUI.api_breaker)
    
    for(let x = 0; x < challenge.data.game_data.waves; x++) {
        fs.writeFileSync(`${x}.gif`, await challenge.getImage())
        console.log(await challenge.answer(parseInt(await ask("Answer: "))))
    }
    console.log("Done!")
})
```


## Support Me
Care to support my work?
* BTC: 38pbL2kX2f6oXGVvc6WFF2BY9VpUCLH7FG
* LTC: M81EXhLSRXNKigqNuz5r7nanAvXmJmjFht
* XRP: rw2ciyaNshpHe7bCHo4bRWq6pqqynnWKQg:865667163
* Ko-Fi (PayPal): [noahcoolboy](https://ko-fi.com/noahcoolboy)
