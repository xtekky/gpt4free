(async () => {

    let response = await fetch("https://play.vercel.ai/openai.jpeg", {
        "headers": {
            "accept": "*/*",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "sec-ch-ua": "\"Chromium\";v=\"112\", \"Google Chrome\";v=\"112\", \"Not:A-Brand\";v=\"99\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
        },
        "referrer": "https://play.vercel.ai/",
        "referrerPolicy": "strict-origin-when-cross-origin",
        "body": null,
        "method": "GET",
        "mode": "cors",
        "credentials": "omit"
    });


    let data = JSON.parse(atob(await response.text()))
    let ret = eval("(".concat(data.c, ")(data.a)"));
    
        botPreventionToken = btoa(JSON.stringify({
        r: ret,
        t: data.t
    }))

    console.log(botPreventionToken);
    
})()