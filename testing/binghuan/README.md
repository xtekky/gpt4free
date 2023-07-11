https://github.com/xtekky/gpt4free/issues/40#issuecomment-1630946450
flow chat process is realy like real Bing (create conversation,listern to websocket and more)
so i just use code Bing Provider from https://gitler.moe/g4f/gpt4free/ version and replace API endpoint and some conversationstyles and work fine

but bing dont realy support multi/continues conversation (using prompt template from original Provider : def convert(messages) : https://github.com/xtekky/gpt4free/blob/e594500c4e7a8443e9b3f4af755c72f42dae83f0/g4f/Provider/Providers/Bing.py#L322)

also i have problem with emoji encoding idk how to fix that