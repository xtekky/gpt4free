audio_models = ['openai-audio', 'openai-audio-large']
image_models = ['dall-e-3', 'gpt-image', 'sdxl-turbo', 'sd-3.5-large', 'flux', 'flux-pro', 'flux-dev', 'flux-schnell', 'flux-redux', 'flux-depth', 'flux-canny', 'flux-kontext', 'flux-dev-lora', 'gpt-image', 'sana', 'gemini-3.5-flash', 'qwen3.7-plus', 'qwen3.7-max', 'qwen3.6-plus', 'qwen3.6-27b', 'qwen3.5-plus', 'qwen3.5-omni-plus', 'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b', 'qwen3.5-omni-flash', 'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus', 'qwen3-vl-plus', 'qwen3-omni-flash-2025-12-01', 'black-forest-labs/FLUX-2-klein-4b', 'black-forest-labs/FLUX-2-klein-9b', 'flux-2-klein-4b', 'flux-2-klein-9b', 'black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell', 'krea/Krea-2-Turbo', 'krea/Krea-2-Raw', 'AlperKTS/Krea2_FP8', 'ideogram-ai/ideogram-4-fp8', 'Tongyi-MAI/Z-Image-Turbo', 'vantagewithai/Krea-2-Turbo-GGUF', 'ostris/ideogram_4_turbotime_lora', 'krea/Krea-2-LoRA-retroanime', 'ideogram-ai/ideogram-4-nf4', 'Phr00t/Qwen-Image-Edit-Rapid-AIO', 'gokaygokay/Krea-2-Realism-LoRA', 'ponpoke/flux2-klein-9b-uncensored-text-encoder', 'flux-dev', 'flux-schnell', 'krea-2-turbo', 'krea-2-raw', 'krea2.fp8', 'ideogram-4', 'z-image-turbo', 'krea-2-turbo-gguf', 'ideogram.4.turbotime.lora', 'krea-2-lora-retroanime', 'ideogram-4-nf4', 'qwen-image-edit-rapid-aio', 'krea-2-realism-lora', 'flux2-klein-9b-uncensored-text-encoder', 'krea/Krea-2-Turbo', 'black-forest-labs/FLUX.1-dev', 'ideogram-ai/ideogram-4-fp8', 'Tongyi-MAI/Z-Image-Turbo', 'ostris/ideogram_4_turbotime_lora', 'krea/Krea-2-LoRA-retroanime', 'black-forest-labs/FLUX.1-schnell', 'stabilityai/stable-diffusion-xl-base-1.0', 'krea/Krea-2-LoRA-darkbrush', 'krea/Krea-2-LoRA-sunsetblur', 'krea/Krea-2-LoRA-neondrip', 'krea/Krea-2-LoRA-vintagetarot', 'Qwen/Qwen-Image', 'baidu/ERNIE-Image', 'ostris/ideogram_4_unconditional_lora', 'krea/Krea-2-LoRA-dotmatrix', 'krea/Krea-2-LoRA-softwatercolor', 'krea/Krea-2-LoRA-kidsdrawing', 'victor/Krea-2-LoRA-magritte', 'Tongyi-MAI/Z-Image', 'krea/Krea-2-LoRA-rainywindow', 'ostris/krea2_turbo_training_adapter', 'baidu/ERNIE-Image-Turbo', 'stabilityai/stable-diffusion-3.5-medium', 'DeverStyle/Ideogram-4.0-Loras', 'fal/FLUX.2-dev-Turbo', 'Qwen/Qwen-Image-2512', 'AIImageStudio/RadianceChromeVoluptuous_Krea2Turbo_v1.0', 'stabilityai/stable-diffusion-3.5-large', 'jmanhype/VHS-Rally-95-LoRA-v1-Ideogram-v4', 'nerijs/pixel-art-xl', 'AunyMoons/loras-pack', 'stabilityai/stable-diffusion-3-medium', 'stabilityai/stable-diffusion-3.5-large-turbo', 'ostris/zimage_turbo_training_adapter', 'jarod2212/Aetheria_Moonlight_Shadow', 'jarod2212/PrimalFashion_SaharaBloom', 'jarod2212/Primal_Fashion_Egyptian_Muse', 'DeverStyle/Krea2-Loras', 'ntc-ai/SDXL-LoRA-slider.great-lighting', 'ntc-ai/SDXL-LoRA-slider.cinematic-lighting', 'Blib-la/caricature_lora_sdxl', 'MarkBW/hannahowo-xl', 'MarkBW/vault-suit-pony-xl', 'MarkBW/olivia-casta-xl', 'MarkBW/smoking-xl', 'MarkBW/knit-sweaterdress-xl', 'MarkBW/deep-neckline-xl', 'MarkBW/sofilein-xl', 'MarkBW/elizabeth-lauren-xl', 'jasperai/flash-sdxl', 'MarkBW/robot-torso-xl', 'MarkBW/emily-bloom2-xl', 'MarkBW/skin-freckles-xl', 'MarkBW/mia-khalifa-xl', 'MarkBW/breckie-hill-xl', 'MarkBW/lily-brown-xl', 'MarkBW/handbra-xl', 'fofr/sdxl-emoji', 'MarkBW/polaroid-filmstyle-xl', 'MarkBW/kaellyn-xl', 'MarkBW/phone-exposure-xl', 'MarkBW/sarang-xl', 'MarkBW/neon-gothic-xl', 'MarkBW/berry0314-pony-xl', 'MarkBW/berry0314-xl', 'jack1101/kpopiu', 'busetolunay/building_flux_lora_v1', 'Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design', 'furaidosu/flux-lora-simple-vector', 'gokaygokay/Flux-White-Background-LoRA', 'Shakker-Labs/FLUX.1-dev-LoRA-Text-Poster', 'gokaygokay/Flux-Double-Exposure-LoRA', 'prithivMLmods/Retro-Pixel-Flux-LoRA', 'gokaygokay/Flux-Realistic-Backgrounds-LoRA', 'strangerzonehf/Flux-Midjourney-Mix2-LoRA', 'data-is-better-together/open-image-preferences-v1-flux-dev-lora', 'mrcuddle/live2d-model-maker', 'AiWise/Detail-Tweaker-XL_v1', 'WiroAI/GTA6-style-flux-lora', 'Keltezaa/katherine-heigl-90s-flux', 'Keltezaa/kate-beckinsale-flux', 'Jonjew/MelissaBenoist', 'Jonjew/MaureenMcCormickMarciaBrady', 'Grahcev/juliette_style_LoRA', 'Jonjew/EvanRachelWood', 'shreenithi20/flux_lora_sketch_style', 'Seanwang1221/Dilraba_FLUX', 'Seanwang1221/Yangmi_SD15_FLUX', 'Rempeyek/Woman_Cloth_XL', 'gokaygokay/Flux-Krea-Realism-LoRA', 'rzgar/fortnite_style_flux_kontext', 'lightx2v/Qwen-Image-Lightning', 'starsfriday/Qwen-Image-EVA-LoRA', 'Muapi/flux-frank-frazetta-style-oil-painting', 'Muapi/game-assets-cartoon-style-3d-isometric-background-assets-for-small-games-flux', 'Muapi/toonfusion-style-ilxl-ponyxl-flux', 'Muapi/iconic-jessica-rabbit', 'Shakker-Labs/AWPortrait-QW', 'Edweibin/flux-dev-nfsw', 'Muapi/clash-royale-style-flux-lora', 'ostris/qwen_image_detail_slider', 'briaai/FIBO', 'Manishsahu53/flux-kontext-fashion-extractor', 'ApathyGhost/EmoElin', 'tarn59/pixel_art_style_lora_z_image_turbo', 'DeverStyle/Z-Image-loras', 'cahlen/black-metal-art-sdxl-lora', 'mikkoph/mikkoph-zimage-turbo', 'NucleusAI/Nucleus-Image', 'Heouzen/LoKR_mc_woman_FLUX1D', 'Bruece/FLUX.1-dev-CMO', 'Muapi/sxz-3d-render-plastic-shader-flux', 'Muapi/fuko-from-undead-unluck', 'Herthu99/Mikahhzimg', 'jarod2212/NeuroSpin_ZIT', 'Muapi/envy-flux-toon-01', 'Muapi/nobody_7-pale-female-uncensored-flux1.d', 'Muapi/game-icon', 'jarod2212/RebelEssence_Grunge_Edition', 'jmanhype/Ektachrome-LoRA-v1-Ideogram-v4', 'Muapi/final-fantasy-concept-art-style-flux', 'Muapi/flux_3d-cartoon', 'Danish1212/pakistani-truck-art-sdxl-lora', 'jarod2212/Rebel_Essence_Emo_Version', 'jarod2212/Rebel_Essence_Bikers', 'Muapi/picture-story-book-for-children-xl-xl', 'Muapi/game-casual-characters', 'Muapi/backtoon-simple-cartoon-background-maker-lora-for-sdxl', 'jarod2212/PrimalFashion_Bailarinas', 'RudySen/Krea2-realism-V1', 'Muapi/china-ktv-girls-ktv', 'mikkoph/mikkoph-krea2', 'ilkerzgi/krea-2-bold-brushy-kidlit-lora', 'ilkerzgi/krea-2-bold-clay-toy-render-lora', 'ilkerzgi/krea-2-chunky-crayon-kidlit-lora', 'ilkerzgi/krea-2-dark-medieval-inkline-lora', 'ilkerzgi/krea-2-dark-ornate-linework-lora', 'ilkerzgi/krea-2-eerie-amber-gloom-lora', 'ilkerzgi/krea-2-engraved-chiaroscuro-ember-lora', 'ilkerzgi/krea-2-felted-wool-whimsy-lora', 'ilkerzgi/krea-2-embroidered-textile-diorama-lora', 'ilkerzgi/krea-2-felt-craft-diorama-lora', 'ilkerzgi/krea-2-electric-blue-horror-poster-lora', 'ilkerzgi/krea-2-felt-craft-miniature-lora', 'ilkerzgi/krea-2-graphic-novel-earthy-ink-lora', 'ilkerzgi/krea-2-indigo-dusk-inkline-lora', 'openfree/flux-chatgpt-ghibli-lora', 'lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1', 'starsfriday/Qwen-Image-NSFW', 'thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO', 'Baraje/SexGod_NSFW_Female_Nudes_QWEN_Image_Edit_2511', 'krea-2-turbo', 'flux-dev', 'ideogram-4', 'z-image-turbo', 'ideogram.4.turbotime.lora', 'krea-2-lora-retroanime', 'flux-schnell', 'stable-diffusion-xl-base-1.0', 'krea-2-lora-darkbrush', 'krea-2-lora-sunsetblur', 'krea-2-lora-neondrip', 'krea-2-lora-vintagetarot', 'qwen-image', 'ernie-image', 'ideogram.4.unconditional.lora', 'krea-2-lora-dotmatrix', 'krea-2-lora-softwatercolor', 'krea-2-lora-kidsdrawing', 'krea-2-lora-magritte', 'z-image', 'krea-2-lora-rainywindow', 'krea2.turbo.training.adapter', 'ernie-image-turbo', 'stable-diffusion-3.5-medium', 'ideogram-4.0-loras', 'flux.2-dev-turbo', 'qwen-image-2512', 'radiancechromevoluptuous.krea2turbo.v1.0', 'sd-3.5-large', 'vhs-rally-95-lora-v1-ideogram', 'pixel-art-xl', 'loras-pack', 'stable-diffusion-3-medium', 'sd-3.5-large-turbo', 'zimage.turbo.training.adapter', 'aetheria.moonlight.shadow', 'primalfashion.saharabloom', 'primal.fashion.egyptian.muse', 'krea2-loras', 'sdxl-lora-slider.great-lighting', 'sdxl-lora-slider.cinematic-lighting', 'caricature.lora.sdxl', 'hannahowo-xl', 'vault-suit-pony-xl', 'olivia-casta-xl', 'smoking-xl', 'knit-sweaterdress-xl', 'deep-neckline-xl', 'sofilein-xl', 'elizabeth-lauren-xl', 'flash-sdxl', 'robot-torso-xl', 'emily-bloom2-xl', 'skin-freckles-xl', 'mia-khalifa-xl', 'breckie-hill-xl', 'lily-brown-xl', 'handbra-xl', 'sdxl-emoji', 'polaroid-filmstyle-xl', 'kaellyn-xl', 'phone-exposure-xl', 'sarang-xl', 'neon-gothic-xl', 'berry0314-pony-xl', 'berry0314-xl', 'kpopiu', 'building.flux.lora.v1', 'flux-dev-lora-logo-design', 'flux-lora-simple-vector', 'flux-white-background-lora', 'flux-dev-lora-text-poster', 'flux-double-exposure-lora', 'retro-pixel-flux-lora', 'flux-realistic-backgrounds-lora', 'flux-midjourney-mix2-lora', 'open-image-preferences-v1-flux-dev-lora', 'live2d-model-maker', 'detail-tweaker-xl.v1', 'gta6-style-flux-lora', 'katherine-heigl-90s-flux', 'kate-beckinsale-flux', 'melissabenoist', 'maureenmccormickmarciabrady', 'juliette.style.lora', 'evanrachelwood', 'flux.lora.sketch.style', 'dilraba.flux', 'yangmi.sd15.flux', 'woman.cloth.xl', 'flux-krea-realism-lora', 'fortnite.style.flux.kontext', 'qwen-image-lightning', 'qwen-image-eva-lora', 'flux-frank-frazetta-style-oil-painting', 'game-assets-cartoon-style-3d-isometric-background-assets-for-small-games-flux', 'toonfusion-style-ilxl-ponyxl-flux', 'iconic-jessica-rabbit', 'awportrait-qw', 'flux-dev-nfsw', 'clash-royale-style-flux-lora', 'qwen-.image.detail.slider', 'fibo', 'flux-kontext-fashion-extractor', 'emoelin', 'pixel.art.style.lora.z.image.turbo', 'z-image-loras', 'black-metal-art-sdxl-lora', 'mikkoph-zimage-turbo', 'nucleus-image', 'lokr.mc.woman.flux1d', 'flux-dev-cmo', 'sxz-3d-render-plastic-shader-flux', 'fuko-from-undead-unluck', 'mikahhzimg', 'neurospin.zit', 'envy-flux-toon-01', 'nobody.7-pale-female-uncensored-flux1.d', 'game-icon', 'rebelessence.grunge.edition', 'ektachrome-lora-v1-ideogram', 'final-fantasy-concept-art-style-flux', 'flux.3d-cartoon', 'pakistani-truck-art-sdxl-lora', 'rebel.essence.emo.version', 'rebel.essence.bikers', 'picture-story-book-for-children-xl-xl', 'game-casual-characters', 'backtoon-simple-cartoon-background-maker-lora-for-sdxl', 'primalfashion.bailarinas', 'krea2-realism', 'china-ktv-girls-ktv', 'mikkoph-krea2', 'krea-2-bold-brushy-kidlit-lora', 'krea-2-bold-clay-toy-render-lora', 'krea-2-chunky-crayon-kidlit-lora', 'krea-2-dark-medieval-inkline-lora', 'krea-2-dark-ornate-linework-lora', 'krea-2-eerie-amber-gloom-lora', 'krea-2-engraved-chiaroscuro-ember-lora', 'krea-2-felted-wool-whimsy-lora', 'krea-2-embroidered-textile-diorama-lora', 'krea-2-felt-craft-diorama-lora', 'krea-2-electric-blue-horror-poster-lora', 'krea-2-felt-craft-miniature-lora', 'krea-2-graphic-novel-earthy-ink-lora', 'krea-2-indigo-dusk-inkline-lora', 'flux-chatgpt-ghibli-lora', 'flux.lustly.ai.uncensored.nsfw.v1', 'qwen-image-nsfw', 'nsfw-master-z-image-turbo', 'sexgod.nsfw.female.nudes.qwen-.image.edit.2511', 'Max', 'botbot2', 'gemini-3.1-flash-image-preview (nano-banana-2) [web-search]', 'gpt-image-1.5-high-fidelity', 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'nonnas-meatballs-open-weight', 'recraft-v4.1-utility-pro', 'flux-2-pro', 'left-bank', 'flux-2-dev', 'seedream-4.5', 'seedream-5.0-lite', 'recraft-v4.1-pro', 'imagen-4.0-generate-001', 'qwen-image-2512', 'hidream-o1-image', 'krea-2-medium', 'wan2.5-preview', 'wan2.5-t2i-preview', 'gpt-image-1-high-fidelity', 'gpt-image-1', 'recraft-v4', 'wan2.7-image-pro', 'krea-2-large', 'wan2.7-image', 'seedream-3', 'z-image', 'flux-1-kontext-max', 'cosmos3-super', 'flux-1-kontext-pro', 'imagen-3.0-generate-002', 'cosmos3-super-agentic', 'ideogram-v3-quality', 'photon', 'recraft-v3', 'lucid-origin', 'flux-1-kontext-dev', 'harbor', 'gpt-image-2 (medium)', 'thunder', 'flow-state', 'citrus', 'habanero', 'spinosaurus', 'flow-state-2', 'kelly', 'flow-state-3', 'hidream-o1-image-1.5', 'nonnos-meatballs-open-weight', 'greenbean', 'waffle', 'flashfennel', 'itadori-sv1', 'pebble-1', 'phantom_brush', 'pebble-2', 'zen-bear-v3', 'auto-bear-v3', 'hidream-e1.1', 'gcps-fast', 'qwen-image-2.0', 'qwen-image-2.0-pro', 'flashbrown-a', 'hunyuan-image-3.0-fal', 'uni-1.1-max', 'uni-1.1', 'soft-shell', 'flashbrown-b', 'instant-ramen', 'zen-bear-v2', 'text-to-image-autoeval-test', 'chives', 'fennel', 'super-cara', 'frenchfry', 'sungod', 'super-gcp', 'zen-bear', 'shakshouka', 'ellsworth', 'parasaurolophus', 'spectral_ink', 'babylon', 'seededit-3.0', 'dialogue', 'altair', 'king-crab', 'paper-lantern', 'crepe', 'blue-crab', 'fennelbaby', 'caudipteryx', 'reve-v1.1-fast', 'zen-bear-v4', 'mussaurus', 'tyrannosaurus', 'auto-bear-v2', 'jalapeno', 'avalon', 'hotate', 'gemini-2.5-flash-image-preview (nano-banana)', 'archaeopteryx', 'snow-crab', 'grok-imagine-image-quality', 'hunyuan-image-2.1', 'red-rock', 'grok-imagine-image', 'dimetrodon', 'phantom_quill', 'qwen-image-edit', 'wan2.5-i2i-preview', 'imagen-4.0-ultra-generate-001', 'mondrian', 'wan2.6-t2i', 'imagen-4.0-fast-generate-001', 'qwen-image-edit-2511', 'chatgpt-image-latest-high-fidelity (20251216)', 'wan2.6-image', 'max', 'botbot2', 'gemini-3.1-flash-image-preview (nano-banana-2) [web-search]', 'gpt-image-1.5-high-fidelity', 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'nonnas-meatballs-open-weight', 'recraft-v4.1-utility-pro', 'flux-2-pro', 'left-bank', 'flux-2-dev', 'seedream-4.5', 'seedream-5.0-lite', 'recraft-v4.1-pro', 'imagen-4.0-generate', 'qwen-image-2512', 'hidream-o1-image', 'krea-2-medium', 'wan2.5', 'wan2.5-t2i', 'gpt-image-1-high-fidelity', 'gpt-image-1', 'recraft', 'wan2.7-image-pro', 'krea-2-large', 'wan2.7-image', 'seedream-3', 'z-image', 'flux-1-kontext-max', 'cosmos3-super', 'flux-1-kontext-pro', 'imagen-3.0-generate', 'cosmos3-super-agentic', 'ideogram-v3-quality', 'photon', 'recraft', 'lucid-origin', 'flux-1-kontext-dev', 'harbor', 'gpt-image-2 (medium)', 'thunder', 'flow-state', 'citrus', 'habanero', 'spinosaurus', 'flow-state-2', 'kelly', 'flow-state-3', 'hidream-o1-image-1.5', 'nonnos-meatballs-open-weight', 'greenbean', 'waffle', 'flashfennel', 'itadori-sv1', 'pebble-1', 'phantom.brush', 'pebble-2', 'zen-bear', 'auto-bear', 'hidream-e1.1', 'gcps-fast', 'qwen-image-2.0', 'qwen-image-2.0-pro', 'flashbrown-a', 'hunyuan-image-3.0-fal', 'uni-1.1-max', 'uni-1.1', 'soft-shell', 'flashbrown-b', 'instant-ramen', 'zen-bear', 'text-to-image-autoeval-test', 'chives', 'fennel', 'super-cara', 'frenchfry', 'sungod', 'super-gcp', 'zen-bear', 'shakshouka', 'ellsworth', 'parasaurolophus', 'spectral.ink', 'babylon', 'seededit-3.0', 'dialogue', 'altair', 'king-crab', 'paper-lantern', 'crepe', 'blue-crab', 'fennelbaby', 'caudipteryx', 'reve-v1.1-fast', 'zen-bear', 'mussaurus', 'tyrannosaurus', 'auto-bear', 'jalapeno', 'avalon', 'hotate', 'gemini-2.5-flash-image-preview (nano-banana)', 'archaeopteryx', 'snow-crab', 'grok-imagine-image-quality', 'hunyuan-image-2.1', 'red-rock', 'grok-imagine-image', 'dimetrodon', 'phantom.quill', 'qwen-image-edit', 'wan2.5-i2i', 'imagen-4.0-ultra-generate', 'mondrian', 'wan2.6-t2i', 'imagen-4.0-fast-generate', 'qwen-image-edit-2511', 'chatgpt-image-high-fidelity (20251216)', 'wan2.6-image', 'flux-kontext-dev', 'flux', 'sd-3.5-large', 'flux-dev', 'flux-kontext-dev', 'flux', 'sd-3.5-large', 'flux-dev']
vision_models = ['auto', 'gpt-5-2', 'gpt-5-2-instant', 'gpt-5-2-thinking', 'gpt-5-1', 'gpt-5-1-instant', 'gpt-5-1-thinking', 'gpt-5', 'gpt-5-instant', 'gpt-5-thinking', 'gpt-4', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.5', 'gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini', 'o3-mini', 'o3-mini-high', 'o4-mini', 'o4-mini-high', 'openai', 'openai-fast', 'gpt-5.4', 'gpt-5.4-mini', 'openai-large', 'mistral-small-3.2', 'mistral', 'gemini-3-flash', 'gemini', 'gemini-flash-lite-3.1', 'gemini-fast', 'gemma', 'grok', 'grok-4-20-reasoning', 'grok-large', 'gemini-search', 'gemini-search-fast', 'gemini-search-large', 'claude-fast', 'claude', 'claude-opus-4.6', 'claude-opus-4.7', 'claude-large', 'kimi', 'kimi-code', 'gemini-large', 'nova', 'llama-maverick', 'llama-scout', 'minimax', 'mistral-large', 'polly', 'qwen-large', 'qwen-vision', 'qwen-vision-pro', 'step-flash', 'qwen3.7-plus', 'qwen3.6-plus', 'qwen3.6-27b', 'qwen-latest-series-invite-beta-v16', 'qwen3.5-plus', 'qwen3.5-omni-plus', 'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b', 'qwen3.5-omni-flash', 'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus', 'qwen3-vl-plus', 'qwen3-omni-flash-2025-12-01', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'Qwen/Qwen2-VL-7B-Instruct', 'llama-3.2-11b-vision', 'qwen-2-vl-7b', 'Max', 'claude-opus-4-6-thinking', 'claude-opus-4-7-thinking', 'claude-opus-4-6', 'claude-opus-4-7', 'gemini-3.1-pro-preview', 'gemini-3.1-pro-preview', 'gpt-5.4-high-no-system-prompt', 'gpt-5.2-chat-latest', 'grok-4.20-beta-0309-reasoning', 'gemini-3-flash', 'gpt-5.5-instant', 'grok-4.20-multi-agent-beta-0309', 'claude-sonnet-4-6', 'gpt-5.4-no-system-prompt', 'gpt-5.1-high', 'minimax-m3', 'minimax-m3', 'gpt-5.4-mini-high', 'gemini-2.5-pro', 'gpt-5.1', 'gpt-5.2-high', 'gpt-5.2', 'gpt-5-high', 'gemini-3.1-flash-lite-preview', 'mimo-v2-omni', 'kimi-k2.5-instant', 'o3-2025-04-16', 'gpt-5-chat', 'qwen3.5-122b-a10b', 'mistral-large-3', 'qwen3-vl-235b-a22b-instruct', 'gpt-4.1-2025-04-14', 'gemini-2.5-flash', 'mistral-medium-2508', 'qwen3.5-27b', 'gpt-5.4-nano-high', 'qwen3.5-flash', 'hunyuan-vision-1.5-thinking', 'qwen3.5-35b-a3b', 'qwen3-vl-235b-a22b-thinking', 'gpt-5-mini-high', 'o4-mini-2025-04-16', 'mistral-medium-2505', 'gpt-4.1-mini-2025-04-14', 'gemma-3-27b-it', 'gemini-2.0-flash-001', 'mistral-small-2506', 'gpt-5-nano-high', 'mistral-small-3.1-24b-instruct-2503', 'steed-0217', 'step-3.7-flash', 'maylynx-alpha', 'may-beta', 'dola-seed-2.0-preview-vision', 'gemma-4-31b-it', 'gpt-5-high-no-system-prompt', 'pisces-0226d', 'pisces-0318-vision', 'june-alpha', 'maymo-beta', 'pisces-0309-vision', 'anonymous-1111', 'spark', 'hearth', 'pisces-0309b', 'scorch', 'dola-seed-2.0-pro-vision', 'nightride-on', 'nightride-on-v2', 'zephyr', 'pisces-0309c', 'atlas', 'vortex', 'botbot2', 'orion', 'pisces-0320', 'kimi-k2.7-code', 'vierra', 'rotten-apple', 'pisces-0309d', 'queen-bee', 'EB45-vision', 'velo', 'kiwi-do', 'mistral-medium-3.5', 'muse-spark', 'stephen-vision-csfix', 'raptor-1123', 'raptor-1124', 'step-3-mini-2511', 'ernie-exp-vl-251016', 'pteronura', 'significant-otter', 'gpt-5.5-xhigh', 'steed-0611', 'kimi-k2.5', 'ernie-5.0-preview-1220', 'momoda-alpha', 'momoda-beta', 'gpt-5-high-new-system-prompt', 'qwen3-vl-8b-thinking', 'kimi-k2.6', 'claude-fable-5', 'qwen3.7-plus', 'steed-0507', 'grok-4.3-high', 'claude-opus-4-8-thinking', 'gemini-3-flash (thinking-minimal)', 'qwen3.5-397b-a17b', 'qwen-vl-max-2025-08-13', 'qwen3-omni-flash', 'qwen3-vl-8b-instruct', 'gpt-5.5-dlp-test', 'grok-4.3', 'may-alpha', 'glassy_lagoon', 'emerald_lagoon', 'gpt-5.5', 'mimo-v2.5', 'claude-opus-4-8', 'amazon.nova-pro-v1:0', 'gpt-5.5-high', 'gemini-3.5-flash', 'glm-5v-turbo', 'kimi-k2.7-code', 'gpt-5.3-codex', 'qwen3.5-122b-a10b-code', 'qwen3.5-27b-code', 'qwen3.5-35b-a3b-code', 'may-alpha', 'lucario', 'eevee', 'gpt-5.4', 'gpt-5.4', 'gpt-5.4-medium', 'gpt-5.2-high', 'gpt-5.4-high', 'metagross', 'riolu', 'gpt-5.5', 'momoda-alpha', 'muse-spark', 'gpt-5.5-xhigh', 'qwen3.7-plus', 'gpt-5.4-high', 'kimi-k2.6', 'gpt-5.5-high', 'qwen3.6-plus', 'gpt-5.4-medium', 'gemini-3.1-flash-image-preview (nano-banana-2) [web-search]', 'gpt-image-1.5-high-fidelity', 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'flux-2-pro', 'flux-2-dev', 'seedream-4.5', 'seedream-5.0-lite', 'gpt-image-1-high-fidelity', 'gpt-image-1', 'wan2.7-image-pro', 'wan2.7-image', 'flux-1-kontext-max', 'flux-1-kontext-pro', 'flux-1-kontext-dev', 'gpt-image-2 (medium)', 'flow-state', 'habanero', 'flow-state-2', 'flow-state-3', 'greenbean', 'flashfennel', 'itadori-sv1', 'phantom_brush', 'zen-bear-v3', 'pebble-2', 'pebble-1', 'hidream-e1.1', 'gcps-fast', 'qwen-image-2.0', 'qwen-image-2.0-pro', 'uni-1.1-max', 'uni-1.1', 'instant-ramen', 'zen-bear-v2', 'chives', 'flow-state', 'fennel', 'sungod', 'super-gcp', 'zen-bear', 'spectral_ink', 'seededit-3.0', 'dialogue', 'paper-lantern', 'ellsworth', 'blue-crab', 'kelly', 'fennelbaby', 'reve-v1.1-fast', 'zen-bear-v4', 'jalapeno', 'gpt-image-2 (medium)', 'avalon', 'gemini-2.5-flash-image-preview (nano-banana)', 'grok-imagine-image-quality', 'red-rock', 'zen-bear-v4', 'grok-imagine-image', 'phantom_quill', 'qwen-image-edit', 'wan2.5-i2i-preview', 'qwen-image-edit-2511', 'chatgpt-image-latest-high-fidelity (20251216)', 'wan2.6-image', 'bouncybohr', 'k2', 'model-x-2', 'veo-3.1-audio-1080p', 'veo-3.1-audio', 'veo-3.1-fast-audio', 'veo-3.1-fast-audio-1080p', 'wan2.6-t2v', 'wan2.5-t2v-preview', 'seedance-v1.5-pro', 'kling-2.5-turbo-1080p', 'kling-2.6-pro', 'pixel-parrot', 'ray-3', 'hailuo-2.3', 'hailuo-02-pro', 'seedance-v1-pro', 'hailuo-02-standard', 'hunyuan-video-1.5', 'veo-2', 'kling-v2.1-master', 'ltx-2-19b', 'wan-v2.2-a14b', 'seedance-v1-lite', 'ray2', 'pika-v2.2', 'wan2.7-i2v', 'markhor', 'grok-imagine-video', 'wan2.6-i2v', 'pixverse-v5.6', 'snowflake', 'grok-imagine-video-1.5-preview-720p', 'hailuo-2.3-fast', 'veo-3.1-audio-4k', 'polaris', 'veo-3.1-fast-audio-4k', 'kling-v2.1-standard', 'hailuo-02-fast', 'kling-v3', 'kling-2.6-standard', 'wan2.5-i2v-preview', 'kandinsky-5.0-i2v-pro', 'runway-gen4-turbo', 'max', 'claude-opus-4-6-thinking', 'claude-opus-4-7-thinking', 'claude-opus-4-6', 'claude-opus-4-7', 'gemini-3.1-pro', 'gemini-3.1-pro', 'gpt-5.4-high-no-system-prompt', 'gpt-5.2-chat', 'grok-4.20-beta-0309-reasoning', 'gemini-3-flash', 'gpt-5.5-instant', 'grok-4.20-multi-agent-beta-0309', 'claude-sonnet-4-6', 'gpt-5.4-no-system-prompt', 'gpt-5.1-high', 'minimax-m3', 'minimax-m3', 'gpt-5.4-mini-high', 'gemini-2.5-pro', 'gpt-5.1', 'gpt-5.2-high', 'gpt-5.2', 'gpt-5-high', 'gemini-3.1-flash-lite', 'mimo-v2-omni', 'kimi-k2.5-instant', 'o3', 'gpt-5-chat', 'qwen-3.5-122b-a10b', 'mistral-large-3', 'qwen-3-vl-235b-a22b', 'gpt-4.1', 'gemini-2.5-flash', 'mistral-medium-2508', 'qwen-3.5-27b', 'gpt-5.4-nano-high', 'qwen-3.5-flash', 'hunyuan-vision-1.5-thinking', 'qwen-3.5-35b-a3b', 'qwen-3-vl-235b-a22b-thinking', 'gpt-5-mini-high', 'o4-mini', 'mistral-medium-2505', 'gpt-4.1-mini', 'gemma-3-27b-it', 'gemini-2.0-flash', 'mistral-small-2506', 'gpt-5-nano-high', 'mistral-small-3.1-24b-2503', 'steed-0217', 'step-3.7-flash', 'maylynx-alpha', 'may-beta', 'dola-seed-2.0-preview-vision', 'gemma-4-31b-it', 'gpt-5-high-no-system-prompt', 'pisces-0226d', 'pisces-0318-vision', 'june-alpha', 'maymo-beta', 'pisces-0309-vision', 'anonymous-1111', 'spark', 'hearth', 'pisces-0309b', 'scorch', 'dola-seed-2.0-pro-vision', 'nightride-on', 'nightride-on', 'zephyr', 'pisces-0309c', 'atlas', 'vortex', 'botbot2', 'orion', 'pisces-0320', 'kimi-k2.7-code', 'vierra', 'rotten-apple', 'pisces-0309d', 'queen-bee', 'eb45-vision', 'velo', 'kiwi-do', 'mistral-medium-3.5', 'muse-spark', 'stephen-vision-csfix', 'raptor-1123', 'raptor-1124', 'step-3-mini-2511', 'ernie-exp-vl-251016', 'pteronura', 'significant-otter', 'gpt-5.5-xhigh', 'steed-0611', 'kimi-k2.5', 'ernie-5.0-preview-1220', 'momoda-alpha', 'momoda-beta', 'gpt-5-high-new-system-prompt', 'qwen-3-vl-8b-thinking', 'kimi-k2.6', 'claude-fable-5', 'qwen-3.7-plus', 'steed-0507', 'grok-4.3-high', 'claude-opus-4-8-thinking', 'gemini-3-flash (thinking-minimal)', 'qwen-3.5-397b-a17b', 'qwen-vl-max', 'qwen-3-omni-flash', 'qwen-3-vl-8b', 'gpt-5.5-dlp-test', 'grok-4.3', 'may-alpha', 'glassy.lagoon', 'emerald.lagoon', 'gpt-5.5', 'mimo-v2.5', 'claude-opus-4-8', 'amazon.nova-pro', 'gpt-5.5-high', 'gemini-3.5-flash', 'glm-5v-turbo', 'kimi-k2.7-code', 'gpt-5.3-codex', 'qwen-3.5-122b-a10b-code', 'qwen-3.5-27b-code', 'qwen-3.5-35b-a3b-code', 'may-alpha', 'lucario', 'eevee', 'gpt-5.4', 'gpt-5.4', 'gpt-5.4-medium', 'gpt-5.2-high', 'gpt-5.4-high', 'metagross', 'riolu', 'gpt-5.5', 'momoda-alpha', 'muse-spark', 'gpt-5.5-xhigh', 'qwen-3.7-plus', 'gpt-5.4-high', 'kimi-k2.6', 'gpt-5.5-high', 'qwen-3.6-plus', 'gpt-5.4-medium', 'gemini-3.1-flash-image-preview (nano-banana-2) [web-search]', 'gpt-image-1.5-high-fidelity', 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'flux-2-pro', 'flux-2-dev', 'seedream-4.5', 'seedream-5.0-lite', 'gpt-image-1-high-fidelity', 'gpt-image-1', 'wan2.7-image-pro', 'wan2.7-image', 'flux-1-kontext-max', 'flux-1-kontext-pro', 'flux-1-kontext-dev', 'gpt-image-2 (medium)', 'flow-state', 'habanero', 'flow-state-2', 'flow-state-3', 'greenbean', 'flashfennel', 'itadori-sv1', 'phantom.brush', 'zen-bear', 'pebble-2', 'pebble-1', 'hidream-e1.1', 'gcps-fast', 'qwen-image-2.0', 'qwen-image-2.0-pro', 'uni-1.1-max', 'uni-1.1', 'instant-ramen', 'zen-bear', 'chives', 'flow-state', 'fennel', 'sungod', 'super-gcp', 'zen-bear', 'spectral.ink', 'seededit-3.0', 'dialogue', 'paper-lantern', 'ellsworth', 'blue-crab', 'kelly', 'fennelbaby', 'reve-v1.1-fast', 'zen-bear', 'jalapeno', 'gpt-image-2 (medium)', 'avalon', 'gemini-2.5-flash-image-preview (nano-banana)', 'grok-imagine-image-quality', 'red-rock', 'zen-bear', 'grok-imagine-image', 'phantom.quill', 'qwen-image-edit', 'wan2.5-i2i', 'qwen-image-edit-2511', 'chatgpt-image-high-fidelity (20251216)', 'wan2.6-image', 'bouncybohr', 'k2', 'model-x-2', 'veo-3.1-audio-1080p', 'veo-3.1-audio', 'veo-3.1-fast-audio', 'veo-3.1-fast-audio-1080p', 'wan2.6-t2v', 'wan2.5-t2v', 'seedance-v1.5-pro', 'kling-2.5-turbo-1080p', 'kling-2.6-pro', 'pixel-parrot', 'ray-3', 'hailuo-2.3', 'hailuo-02-pro', 'seedance-v1-pro', 'hailuo-02-standard', 'hunyuan-video-1.5', 'veo-2', 'kling-v2.1-master', 'ltx-2-19b', 'wan-v2.2-a14b', 'seedance-v1-lite', 'ray2', 'pika-v2.2', 'wan2.7-i2v', 'markhor', 'grok-imagine-video', 'wan2.6-i2v', 'pixverse-v5.6', 'snowflake', 'grok-imagine-video-1.5-preview-720p', 'hailuo-2.3-fast', 'veo-3.1-audio-4k', 'polaris', 'veo-3.1-fast-audio-4k', 'kling-v2.1-standard', 'hailuo-02-fast', 'kling', 'kling-2.6-standard', 'wan2.5-i2v', 'kandinsky-5.0-i2v-pro', 'runway-gen4-turbo', 'azure:openai/gpt-4o', 'azure:openai/gpt-5', 'azure:openai/gpt-5-codex', 'azure:openai/gpt-5-mini', 'azure:openai/gpt-5-nano', 'azure:openai/gpt-5.1', 'azure:openai/gpt-5.1-codex', 'azure:openai/gpt-5.1-codex-mini', 'azure:openai/gpt-5.2', 'azure:openai/gpt-5.2-codex', 'azure:openai/gpt-5.3-codex', 'azure:openai/gpt-5.4', 'azure:openai/gpt-5.4-mini', 'azure:openai/gpt-5.4-nano', 'moonshotai:moonshotai/moonshot-v1-128k-vision-preview', 'moonshotai:moonshotai/moonshot-v1-32k-vision-preview', 'moonshotai:moonshotai/moonshot-v1-8k-vision-preview', 'openai:openai/gpt-4.1', 'openai:openai/gpt-4.1-mini', 'openai:openai/gpt-4.1-nano', 'openai:openai/gpt-4.5-preview', 'openai:openai/gpt-4o-mini', 'openai:openai/gpt-5', 'openai:openai/gpt-5-chat', 'openai:openai/gpt-5-mini', 'openai:openai/gpt-5-nano', 'openai:openai/gpt-5.1-chat', 'openai:openai/gpt-5.2', 'openai:openai/gpt-5.2-chat', 'openai:openai/gpt-5.2-pro', 'openai:openai/gpt-5.4', 'openai:openai/gpt-5.4-pro', 'openai:openai/gpt-5.5', 'openai:openai/gpt-5.5-pro', 'openai:openai/o1', 'openai:openai/o1-mini', 'openai:openai/o1-pro', 'openai:openai/o3', 'openai:openai/o3-mini', 'openai:openai/o3-pro', 'openai:openai/o4-mini', 'openrouter:meta-llama/llama-3.2-11b-vision-instruct', 'openrouter:openai/gpt-3.5-turbo', 'openrouter:openai/gpt-3.5-turbo-0613', 'openrouter:openai/gpt-3.5-turbo-16k', 'openrouter:openai/gpt-3.5-turbo-instruct', 'openrouter:openai/gpt-4', 'openrouter:openai/gpt-4-turbo', 'openrouter:openai/gpt-4-turbo-preview', 'openrouter:openai/gpt-4o-2024-05-13', 'openrouter:openai/gpt-4o-2024-08-06', 'openrouter:openai/gpt-4o-2024-11-20', 'openrouter:openai/gpt-4o-mini-2024-07-18', 'openrouter:openai/gpt-4o-mini-search-preview', 'openrouter:openai/gpt-4o-search-preview', 'openrouter:openai/gpt-5-image', 'openrouter:openai/gpt-5-image-mini', 'openrouter:openai/gpt-5-pro', 'openrouter:openai/gpt-5.1-codex-max', 'openrouter:openai/gpt-5.3-chat', 'openrouter:openai/gpt-5.4-image-2', 'openrouter:openai/gpt-audio', 'openrouter:openai/gpt-audio-mini', 'openrouter:openai/gpt-chat-latest', 'openrouter:openai/gpt-oss-120b:free', 'openrouter:openai/gpt-oss-20b:free', 'openrouter:openai/gpt-oss-safeguard-20b', 'openrouter:openai/o3-deep-research', 'openrouter:openai/o3-mini-high', 'openrouter:openai/o4-mini-deep-research', 'openrouter:openai/o4-mini-high', 'openrouter:sao10k/l3-lunaris-8b', 'openrouter:sao10k/l3.1-70b-hanami-x1', 'openrouter:sao10k/l3.1-euryale-70b', 'openrouter:sao10k/l3.3-euryale-70b', 'openrouter:~openai/gpt-latest', 'openrouter:~openai/gpt-mini-latest', 'togetherai:openai/gpt-oss-120b', 'togetherai:openai/gpt-oss-20b', 'x-ai:x-ai/grok-2-vision', 'x-ai:x-ai/grok-2-vision-1212', 'x-ai:x-ai/grok-vision-beta', 'gpt-4o', 'gpt-5', 'gpt-5-codex', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.1-codex', 'gpt-5.1-codex-mini', 'gpt-5.2', 'gpt-5.2-codex', 'gpt-5.3-codex', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'moonshot-v1-128k-vision', 'moonshot-v1-32k-vision', 'moonshot-v1-8k-vision', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.5', 'gpt-4o-mini', 'gpt-5', 'gpt-5-chat', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5.1-chat', 'gpt-5.2', 'gpt-5.2-chat', 'gpt-5.2-pro', 'gpt-5.4', 'gpt-5.4-pro', 'gpt-5.5', 'gpt-5.5-pro', 'o1', 'o1-mini', 'o1-pro', 'o3', 'o3-mini', 'o3-pro', 'o4-mini', 'llama-3.2-11b-vision', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o', 'gpt-4o', 'gpt-4o-mini', 'gpt-4o-mini-search', 'gpt-4o-search', 'gpt-5-image', 'gpt-5-image-mini', 'gpt-5-pro', 'gpt-5.1-codex-max', 'gpt-5.3-chat', 'gpt-5.4-image-2', 'gpt-audio', 'gpt-audio-mini', 'gpt-chat', 'gpt-oss-120b', 'gpt-oss-20b', 'gpt-oss-safeguard-20b', 'o3-deep-research', 'o3-mini-high', 'o4-mini-deep-research', 'o4-mini-high', 'l3-lunaris-8b', 'l3.1-70b-hanami-x1', 'l3.1-euryale-70b', 'l3.3-euryale-70b', 'gpt', 'gpt-mini', 'gpt-oss-120b', 'gpt-oss-20b', 'grok-2-vision', 'grok-2-vision-1212', 'grok-vision-beta']
video_models = ['Wan-AI/Wan2.2-TI2V-5B', 'meituan-longcat/LongCat-Video', 'Wan-AI/Wan2.1-T2V-1.3B', 'Wan-AI/Wan2.1-T2V-14B', 'tencent/HunyuanVideo-1.5', 'Wan-AI/Wan2.2-T2V-A14B', 'Wan-AI/Wan2.2-T2V-A14B-Diffusers', 'zai-org/CogVideoX-5b', 'genmo/mochi-1-preview', 'wan2.2-ti2v-5b', 'longcat-video', 'wan2.1-t2v-1.3b', 'wan2.1-t2v-14b', 'hunyuanvideo-1.5', 'wan2.2-t2v-a14b', 'wan2.2-t2v-a14b-diffusers', 'cogvideox-5b', 'mochi-1', 'bouncybohr', 'k2', 'model-x-2', 'veo-3.1-audio-1080p', 'wan2.7-t2v', 'veo-3.1-audio', 'sora-2-pro', 'veo-3.1-fast-audio', 'veo-3.1-fast-audio-1080p', 'wan2.6-t2v', 'sora-2', 'wan2.5-t2v-preview', 'seedance-v1.5-pro', 'runway-gen-4.5', 'kling-2.5-turbo-1080p', 'kling-2.6-pro', 'pixel-parrot', 'kling-o1-pro', 'ray-3', 'hailuo-2.3', 'hailuo-02-pro', 'seedance-v1-pro', 'hailuo-02-standard', 'kandinsky-5.0-t2v-pro', 'hunyuan-video-1.5', 'veo-2', 'kling-v2.1-master', 'ltx-2-19b', 'wan-v2.2-a14b', 'kandinsky-5.0-t2v-lite', 'seedance-v1-lite', 'sora', 'ray2', 'pika-v2.2', 'mochi-v1', 'tbd', 'wan2.7-i2v', 'markhor', 'grok-imagine-video', 'model-x', 'wan2.6-i2v', 'runway-gen4-aleph', 'pixverse-v5.6', 'snowflake', 'grok-imagine-video-1.5-preview-720p', 'bubble-tea', 'hailuo-2.3-fast', 'kling-o3-pro', 'culumus', 'veo-3.1-audio-4k', 'polaris', 'veo-3.1-fast-audio-4k', 'kling-v3', 'kling-v2.1-standard', 'hailuo-02-fast', 'kling-2.6-standard', 'wan2.5-i2v-preview', 'wan-vace', 'kandinsky-5.0-i2v-pro', 'runway-gen4-turbo', 'bouncybohr', 'k2', 'model-x-2', 'veo-3.1-audio-1080p', 'wan2.7-t2v', 'veo-3.1-audio', 'sora-2-pro', 'veo-3.1-fast-audio', 'veo-3.1-fast-audio-1080p', 'wan2.6-t2v', 'sora-2', 'wan2.5-t2v', 'seedance-v1.5-pro', 'runway-gen-4.5', 'kling-2.5-turbo-1080p', 'kling-2.6-pro', 'pixel-parrot', 'kling-o1-pro', 'ray-3', 'hailuo-2.3', 'hailuo-02-pro', 'seedance-v1-pro', 'hailuo-02-standard', 'kandinsky-5.0-t2v-pro', 'hunyuan-video-1.5', 'veo-2', 'kling-v2.1-master', 'ltx-2-19b', 'wan-v2.2-a14b', 'kandinsky-5.0-t2v-lite', 'seedance-v1-lite', 'sora', 'ray2', 'pika-v2.2', 'mochi', 'tbd', 'wan2.7-i2v', 'markhor', 'grok-imagine-video', 'model-x', 'wan2.6-i2v', 'runway-gen4-aleph', 'pixverse-v5.6', 'snowflake', 'grok-imagine-video-1.5-preview-720p', 'bubble-tea', 'hailuo-2.3-fast', 'kling-o3-pro', 'culumus', 'veo-3.1-audio-4k', 'polaris', 'veo-3.1-fast-audio-4k', 'kling', 'kling-v2.1-standard', 'hailuo-02-fast', 'kling-2.6-standard', 'wan2.5-i2v', 'wan-vace', 'kandinsky-5.0-i2v-pro', 'runway-gen4-turbo', 'video']
model_map = {
  "default": {
    "Ollama": "",
    "Qwen": "",
    "DeepInfra": "",
    "PollinationsAI": "",
    "WeWordle": "",
    "GLM": "",
    "TeachAnything": "",
    "OpenaiChat": "",
    "Together": "",
    "OperaAria": "",
    "CopilotApp": ""
  },
  "gpt-4": {
    "CopilotApp": "chat",
    "Yqcloud": "gpt-4",
    "WeWordle": "gpt-4",
    "OpenaiChat": "gpt-4",
    "Copilot": "Copilot",
    "CopilotAccount": "Copilot",
    "PuterJS": "openrouter:openai/gpt-4",
    "ApiAirforce": "gpt-4",
    "CopilotSession": "Copilot",
    "OpenRouter": "openai/gpt-4"
  },
  "gpt-4o": {
    "CopilotApp": "chat",
    "OpenaiChat": "gpt-4o",
    "Copilot": "Copilot",
    "CopilotAccount": "Copilot",
    "PuterJS": "openrouter:openai/gpt-4o-2024-11-20",
    "CopilotSession": "Copilot",
    "OpenRouter": "openai/gpt-4o",
    "WeWordle": "gpt-4o"
  },
  "gpt-4o-mini": {
    "OpenaiChat": "gpt-4o-mini",
    "PuterJS": "openrouter:openai/gpt-4o-mini-2024-07-18",
    "ApiAirforce": "gpt-4o-mini",
    "OpenRouter": "openai/gpt-4o-mini-2024-07-18",
    "WeWordle": "gpt-4o-mini",
    "Surfsense": "gpt-o4-mini-no-login"
  },
  "gpt-4o-mini-tts": {
    "OpenAIFM": "coral",
    "ApiAirforce": "gpt-4o-mini-tts"
  },
  "o1": {
    "OpenaiAccount": "o1",
    "OpenaiChat": "o1",
    "Copilot": "Think Deeper",
    "CopilotAccount": "Think Deeper",
    "PuterJS": "openai:openai/o1",
    "ApiAirforce": "o1",
    "CopilotApp": "reasoning",
    "CopilotSession": "Think Deeper",
    "OpenRouter": "openai/o1"
  },
  "o1-mini": {
    "OpenaiAccount": "o1-mini",
    "OpenaiChat": "o1-mini",
    "PuterJS": "openai:openai/o1-mini"
  },
  "o3-mini": {
    "OpenaiChat": "o3-mini",
    "LMArena": "o3-mini",
    "PuterJS": "openai:openai/o3-mini",
    "ApiAirforce": "o3-mini-2025-01-31",
    "CopilotApp": "reasoning",
    "OpenRouter": "openai/o3-mini"
  },
  "o3-mini-high": {
    "OpenaiAccount": "o3-mini-high",
    "OpenaiChat": "o3-mini-high",
    "PuterJS": "openrouter:openai/o3-mini-high",
    "OpenRouter": "openai/o3-mini-high"
  },
  "o4-mini": {
    "OpenaiChat": "o4-mini",
    "LMArena": "o4-mini-2025-04-16",
    "PuterJS": "openai:openai/o4-mini",
    "ApiAirforce": "o4-mini",
    "OpenRouter": "openai/o4-mini",
    "Surfsense": "gpt-o4-mini-no-login"
  },
  "o4-mini-high": {
    "OpenaiChat": "o4-mini-high",
    "PuterJS": "openrouter:openai/o4-mini-high",
    "OpenRouter": "openai/o4-mini-high"
  },
  "gpt-4.1": {
    "OpenaiChat": "gpt-4-1",
    "LMArena": "gpt-4.1-2025-04-14",
    "PuterJS": "openai:openai/gpt-4.1",
    "ApiAirforce": "gpt-4.1",
    "OpenRouter": "openai/gpt-4.1"
  },
  "gpt-4.1-mini": {
    "OpenaiChat": "gpt-4-1-mini",
    "LMArena": "gpt-4.1-mini-2025-04-14",
    "PuterJS": "openai:openai/gpt-4.1-mini",
    "OpenRouter": "openai/gpt-4.1-mini"
  },
  "gpt-4.1-nano": {
    "PollinationsAI": "openai-fast",
    "PuterJS": "openai:openai/gpt-4.1-nano",
    "ApiAirforce": "gpt-4.1-nano",
    "OpenRouter": "openai/gpt-4.1-nano"
  },
  "gpt-4.5": {
    "OpenaiChat": "gpt-4-5",
    "PuterJS": "openai:openai/gpt-4.5-preview"
  },
  "gpt-4.5-mini": {
    "Surfsense": "gpt-5.4-mini-no-login"
  },
  "gpt-oss-120b": {
    "Together": "openai/gpt-oss-120b",
    "HuggingFace": "openai/gpt-oss-120b",
    "OpenRouter": "openai/gpt-oss-120b",
    "Groq": "openai/gpt-oss-120b",
    "LMArena": "gpt-oss-120b",
    "PuterJS": "togetherai:openai/gpt-oss-120b",
    "ApiAirforce": "gpt-oss-120b",
    "HuggingChat": "openai/gpt-oss-120b",
    "HuggingFaceAPI": "openai/gpt-oss-120b",
    "Ollama": "gpt-oss:120b"
  },
  "smart": {
    "CopilotApp": "smart",
    "CopilotAccount": "smart",
    "Copilot": "smart"
  },
  "reasoning": {
    "CopilotApp": "reasoning",
    "CopilotAccount": "reasoning"
  },
  "study": {
    "CopilotApp": "study",
    "CopilotAccount": "Study",
    "Copilot": "Study",
    "CopilotSession": "Study"
  },
  "search": {
    "CopilotApp": "search",
    "CopilotAccount": "search",
    "Video": "search"
  },
  "dall-e-3": {
    "OpenaiAccount": "dall-e-3",
    "MicrosoftDesigner": "dall-e-3",
    "BingCreateImages": "dall-e-3",
    "CopilotAccount": "Copilot",
    "ApiAirforce": "dall-e-3",
    "OpenaiChat": "gpt-image"
  },
  "gpt-image": {
    "PollinationsImage": "gpt-image",
    "OpenaiChat": "gpt-image",
    "PollinationsAI": "gptimage"
  },
  "meta-ai": {
    "MetaAI": "meta-ai",
    "MetaAIAccount": "meta-ai"
  },
  "llama-2-70b": {
    "Together": "llama-2-70b",
    "PuterJS": "openrouter:meta-llama/llama-2-70b-chat"
  },
  "llama-3-8b": {
    "Together": "llama-3-8b",
    "PuterJS": "openrouter:meta-llama/llama-3-8b-instruct",
    "HuggingChat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "HuggingFaceAPI": "meta-llama/Meta-Llama-3-8B-Instruct",
    "OpenRouter": "meta-llama/llama-3-8b-instruct",
    "PerplexityApi": "llama-3-8b-instruct"
  },
  "llama-3-70b": {
    "Together": "llama-3-70b",
    "PuterJS": "openrouter:meta-llama/llama-3-70b-instruct",
    "HuggingFaceAPI": "meta-llama/Meta-Llama-3-70B-Instruct",
    "PerplexityApi": "llama-3-70b-instruct",
    "Replicate": "meta/meta-llama-3-70b-instruct"
  },
  "llama-3.1-8b": {
    "Together": "llama-3.1-8b",
    "HuggingFace": "meta-llama/Llama-3.1-8B",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.1-8b-instruct:free",
      "openrouter:meta-llama/llama-3.1-8b-instruct"
    ],
    "Cerebras": "llama3.1-8b",
    "GlhfChat": "hf:meta-llama/Llama-3.1-8B-Instruct",
    "HuggingChat": "meta-llama/Llama-3.1-8B-Instruct",
    "HuggingFaceAPI": "meta-llama/Llama-3.1-8B-Instruct",
    "OpenRouter": "meta-llama/llama-3.1-8b-instruct"
  },
  "llama-3.1-70b": {
    "Together": "llama-3.1-70b",
    "PuterJS": "openrouter:meta-llama/llama-3.1-70b-instruct",
    "Cerebras": "llama3.1-70b",
    "GlhfChat": "hf:meta-llama/Llama-3.1-70B-Instruct",
    "HuggingChat": "meta-llama/Llama-3.1-70B-Instruct",
    "OpenRouter": "meta-llama/llama-3.1-70b-instruct"
  },
  "llama-3.1-405b": {
    "Together": "llama-3.1-405b",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.1-405b:free",
      "openrouter:meta-llama/llama-3.1-405b",
      "openrouter:meta-llama/llama-3.1-405b-instruct"
    ],
    "GlhfChat": "hf:meta-llama/Llama-3.1-405B-Instruct"
  },
  "llama-3.2-3b": {
    "Together": "llama-3.2-3b",
    "HuggingFace": "meta-llama/Llama-3.2-3B-Instruct",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.2-3b-instruct:free",
      "openrouter:meta-llama/llama-3.2-3b-instruct"
    ],
    "GlhfChat": "hf:meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFaceAPI": "meta-llama/Llama-3.2-3B-Instruct",
    "OpenRouter": "meta-llama/llama-3.2-3b-instruct"
  },
  "llama-3.2-11b": {
    "Together": "llama-3.2-11b",
    "HuggingChat": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "HuggingFace": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.2-11b-vision-instruct:free",
      "openrouter:meta-llama/llama-3.2-11b-vision-instruct"
    ],
    "HuggingFaceAPI": "meta-llama/Llama-3.2-11B-Vision-Instruct"
  },
  "llama-3.2-90b": {
    "Together": "llama-3.2-90b",
    "PuterJS": "openrouter:meta-llama/llama-3.2-90b-vision-instruct"
  },
  "llama-3.3-70b": {
    "Together": "llama-3.3-70b",
    "HuggingChat": "meta-llama/Llama-3.3-70B-Instruct",
    "HuggingFace": "meta-llama/Llama-3.3-70B-Instruct",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.3-70b-instruct:free",
      "openrouter:meta-llama/llama-3.3-70b-instruct"
    ],
    "Cerebras": "llama-3.3-70b",
    "GlhfChat": "hf:meta-llama/Llama-3.3-70B-Instruct",
    "HuggingFaceAPI": "meta-llama/Llama-3.3-70B-Instruct",
    "OpenRouter": "meta-llama/llama-3.3-70b-instruct",
    "PollinationsAI": "llama"
  },
  "llama-4-scout": {
    "PollinationsAI": "llama-scout",
    "Together": "llama-4-scout",
    "PuterJS": [
      "openrouter:meta-llama/llama-4-scout:free",
      "openrouter:meta-llama/llama-4-scout"
    ],
    "OpenRouter": "meta-llama/llama-4-scout"
  },
  "llama-4-maverick": {
    "Together": "llama-4-maverick",
    "PuterJS": [
      "openrouter:meta-llama/llama-4-maverick:free",
      "openrouter:meta-llama/llama-4-maverick"
    ],
    "OpenRouter": "meta-llama/llama-4-maverick",
    "PollinationsAI": "llama-maverick"
  },
  "mistral-7b": {
    "Together": "mistral-7b"
  },
  "mixtral-8x7b": {
    "Together": "mixtral-8x7b"
  },
  "mistral-nemo": {
    "HuggingChat": "mistralai/Mistral-Nemo-Instruct-2407",
    "HuggingFace": "mistralai/Mistral-Nemo-Instruct-2407",
    "HuggingFaceAPI": "mistralai/Mistral-Nemo-Instruct-2407",
    "OllamaSwarm": "mistral-nemo:latest",
    "OpenRouter": "mistralai/mistral-nemo"
  },
  "mistral-small-24b": {
    "Together": "mistral-small-24b"
  },
  "mistral-small-3.1-24b": {
    "PollinationsAI": "mistral-small",
    "PuterJS": "openrouter:mistralai/mistral-small-3.1-24b-instruct",
    "OpenRouter": "mistralai/mistral-small-3.1-24b-instruct"
  },
  "hermes-2-dpo": {
    "Together": "hermes-2-dpo",
    "PuterJS": "openrouter:nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
  },
  "phi-3.5-mini": {
    "HuggingChat": "microsoft/Phi-3.5-mini-instruct",
    "HuggingFace": "microsoft/Phi-3.5-mini-instruct",
    "PuterJS": "openrouter:microsoft/phi-3.5-mini-128k-instruct",
    "HuggingFaceAPI": "microsoft/Phi-3.5-mini-instruct"
  },
  "gemini-2.0": {
    "Gemini": "gemini-2.0"
  },
  "gemini-2.0-flash": {
    "Gemini": "gemini-2.0-flash",
    "GeminiPro": "gemini-2.0-flash",
    "LMArena": "gemini-2.0-flash-001",
    "PuterJS": [
      "gemini-2.0-flash",
      "openrouter:google/gemini-2.0-flash-lite-001",
      "openrouter:google/gemini-2.0-flash-001",
      "openrouter:google/gemini-2.0-flash-exp:free"
    ]
  },
  "gemini-2.0-flash-thinking": {
    "Gemini": "gemini-2.0-flash-thinking",
    "GeminiPro": "gemini-2.0-flash-thinking"
  },
  "gemini-2.0-flash-thinking-with-apps": {
    "Gemini": "gemini-2.0-flash-thinking-with-apps"
  },
  "gemini-2.5-flash": {
    "Gemini": "gemini-2.5-flash",
    "GeminiPro": "gemini-2.5-flash",
    "GeminiCLI": "gemini-2.5-flash",
    "LMArena": "gemini-2.5-flash",
    "PuterJS": "openrouter:google/gemini-2.5-flash-preview",
    "Antigravity": "gemini-2.5-flash",
    "ApiAirforce": "gemini-2.5-flash",
    "OpenRouter": "google/gemini-2.5-flash"
  },
  "gemini-2.5-pro": {
    "Gemini": "gemini-2.5-pro",
    "GeminiPro": "gemini-2.5-pro",
    "GeminiCLI": "gemini-2.5-pro",
    "LMArena": "gemini-2.5-pro",
    "PuterJS": [
      "openrouter:google/gemini-2.5-pro-preview",
      "openrouter:google/gemini-2.5-pro-exp-03-25"
    ],
    "Antigravity": "gemini-2.5-pro",
    "ApiAirforce": "gemini-2.5-pro",
    "OpenRouter": "google/gemini-2.5-pro-preview-05-06",
    "PollinationsAI": "gemini-large"
  },
  "gemini-3-pro-preview": {
    "GeminiCLI": "gemini-3-pro-preview"
  },
  "gemini-3.1-pro": {
    "Gemini": "gemini-3.1-pro",
    "LMArena": "gemini-3.1-pro",
    "PuterJS": "openrouter:google/gemini-3.1-pro-preview",
    "ApiAirforce": "gemini-3.1-pro",
    "OpenRouter": "google/gemini-3.1-pro-preview",
    "PollinationsAI": "gemini-large"
  },
  "gemini-3.1-flash-lite": {
    "Gemini": "gemini-3.1-flash-lite",
    "LMArena": "gemini-3.1-flash-lite-preview",
    "PuterJS": "openrouter:google/gemini-3.1-flash-lite-preview",
    "ApiAirforce": "gemini-3.1-flash-lite",
    "OpenRouter": "google/gemini-3.1-flash-lite-preview",
    "PollinationsAI": "gemini-flash-lite-3.1"
  },
  "gemini-3.5-flash": {
    "Gemini": "gemini-3.5-flash",
    "LMArena": "gemini-3.5-flash",
    "PuterJS": "openrouter:google/gemini-3.5-flash",
    "ApiAirforce": "gemini-3.5-flash",
    "OpenRouter": "google/gemini-3.5-flash",
    "PollinationsAI": "gemini"
  },
  "gemma-2-27b": {
    "Together": "gemma-2-27b",
    "HuggingFace": "google/gemma-2-27b-it",
    "PuterJS": "openrouter:google/gemma-2-27b-it",
    "HuggingChat": "google/gemma-2-27b-it",
    "HuggingFaceAPI": "google/gemma-2-27b-it"
  },
  "gemma-3-27b": {
    "Together": "gemma-3-27b",
    "PuterJS": [
      "openrouter:google/gemma-3-27b-it:free",
      "openrouter:google/gemma-3-27b-it"
    ]
  },
  "gemma-3n-e4b": {
    "Together": "gemma-3n-e4b"
  },
  "command-r": {
    "HuggingSpace": "command-r-08-2024",
    "PuterJS": [
      "openrouter:cohere/command-r-08-2024",
      "openrouter:cohere/command-r",
      "openrouter:cohere/command-r-03-2024"
    ],
    "CohereForAI_C4AI_Command": "command-r-08-2024",
    "OllamaSwarm": "command-r:35b"
  },
  "command-r-plus": {
    "HuggingSpace": [
      "command-r-plus-08-2024",
      "command-r-plus"
    ],
    "HuggingChat": "CohereForAI/c4ai-command-r-plus-08-2024",
    "HuggingFace": "CohereForAI/c4ai-command-r-plus-08-2024",
    "PuterJS": [
      "openrouter:cohere/command-r-plus-08-2024",
      "openrouter:cohere/command-r-plus",
      "openrouter:cohere/command-r-plus-04-2024"
    ],
    "CohereForAI_C4AI_Command": [
      "command-r-plus-08-2024",
      "command-r-plus"
    ],
    "HuggingFaceAPI": "CohereForAI/c4ai-command-r-plus-08-2024",
    "OllamaSwarm": "command-r-plus:104b"
  },
  "command-r7b": {
    "HuggingSpace": [
      "command-r7b-12-2024",
      "command-r7b-arabic-02-2025"
    ],
    "PuterJS": "openrouter:cohere/command-r7b-12-2024",
    "CohereForAI_C4AI_Command": [
      "command-r7b-12-2024",
      "command-r7b-arabic-02-2025"
    ]
  },
  "command-a": {
    "HuggingSpace": "command-a-03-2025",
    "PuterJS": "openrouter:cohere/command-a",
    "CohereForAI_C4AI_Command": "command-a-03-2025",
    "OpenRouter": "cohere/command-a"
  },
  "qwen-2-72b": {
    "Together": "qwen-2-72b",
    "HuggingFace": "Qwen/Qwen2-72B-Instruct",
    "PuterJS": "openrouter:qwen/qwen-2-72b-instruct",
    "HuggingChat": "Qwen/Qwen2-72B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2-72B-Instruct"
  },
  "qwen-2-vl-7b": {
    "HuggingFaceAPI": "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingFace": "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingChat": "Qwen/Qwen2-VL-7B-Instruct"
  },
  "qwen-2-vl-72b": {
    "Together": "qwen-2-vl-72b"
  },
  "qwen-2.5-7b": {
    "Together": "qwen-2.5-7b",
    "HuggingFace": "Qwen/Qwen2.5-7B-Instruct",
    "PuterJS": [
      "openrouter:qwen/qwen-2.5-7b-instruct:free",
      "openrouter:qwen/qwen-2.5-7b-instruct"
    ],
    "GlhfChat": "hf:Qwen/Qwen2.5-7B-Instruct",
    "HuggingChat": "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2.5-7B",
    "OpenRouter": "qwen/qwen-2.5-7b-instruct"
  },
  "qwen-2.5-72b": {
    "Together": "qwen-2.5-72b",
    "HuggingFace": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "PuterJS": [
      "openrouter:qwen/qwen-2.5-72b-instruct:free",
      "openrouter:qwen/qwen-2.5-72b-instruct"
    ],
    "GlhfChat": "hf:Qwen/Qwen2.5-72B-Instruct",
    "HuggingChat": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "OpenRouter": "qwen/qwen-2.5-72b-instruct"
  },
  "qwen-2.5-coder-32b": {
    "Together": "qwen-2.5-coder-32b",
    "HuggingChat": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "HuggingFace": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "PuterJS": [
      "openrouter:qwen/qwen-2.5-coder-32b-instruct:free",
      "openrouter:qwen/qwen-2.5-coder-32b-instruct"
    ],
    "GlhfChat": "hf:Qwen/Qwen2.5-Coder-32B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "OpenRouter": "qwen/qwen-2.5-coder-32b-instruct",
    "PollinationsAI": "qwen-3-coder"
  },
  "qwen-2.5-vl-72b": {
    "Together": "qwen-2.5-vl-72b",
    "PuterJS": [
      "openrouter:qwen/qwen2.5-vl-72b-instruct:free",
      "openrouter:qwen/qwen2.5-vl-72b-instruct"
    ],
    "HuggingChat": "Qwen/Qwen2.5-VL-72B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2.5-VL-72B-Instruct",
    "OpenRouter": "qwen/qwen2.5-vl-72b-instruct"
  },
  "qwen-3-235b": {
    "Together": "qwen-3-235b",
    "PuterJS": [
      "openrouter:qwen/qwen3-235b-a22b:free",
      "openrouter:qwen/qwen3-235b-a22b"
    ]
  },
  "qwen-3-32b": {
    "Together": "qwen-3-32b",
    "PuterJS": [
      "openrouter:qwen/qwen3-32b:free",
      "openrouter:qwen/qwen3-32b"
    ],
    "ApiAirforce": "qwen3-32b",
    "HuggingChat": "Qwen/Qwen3-32B",
    "HuggingFaceAPI": "Qwen/Qwen3-32B",
    "OpenRouter": "qwen/qwen3-32b"
  },
  "qwq-32b": {
    "Together": "qwq-32b",
    "HuggingChat": "Qwen/QwQ-32B",
    "HuggingFace": "Qwen/QwQ-32B",
    "LMArena": "qwq-32b",
    "PuterJS": [
      "openrouter:qwen/qwq-32b-preview",
      "openrouter:qwen/qwq-32b:free",
      "openrouter:qwen/qwq-32b"
    ],
    "GlhfChat": "hf:Qwen/QwQ-32B-Preview",
    "HuggingFaceAPI": "Qwen/QwQ-32B"
  },
  "deepseek-v3": {
    "Together": "deepseek-v3",
    "PuterJS": "openrouter:deepseek/deepseek-v3-base:free"
  },
  "deepseek-r1": {
    "PollinationsAI": "deepseek-reasoning",
    "Together": "deepseek-r1",
    "HuggingChat": "deepseek-ai/DeepSeek-R1",
    "HuggingFace": "deepseek-ai/DeepSeek-R1",
    "PuterJS": [
      "deepseek-reasoner",
      "openrouter:deepseek/deepseek-r1:free",
      "openrouter:deepseek/deepseek-r1"
    ],
    "ApiAirforce": "deepseek-r1",
    "Cerebras": "deepseek-r1-distill-llama-70b",
    "DeepSeekAPI": "deepseek-r1",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-R1",
    "OllamaSwarm": "deepseek-r1:7b",
    "OpenRouter": "deepseek/deepseek-r1",
    "WeWordle": "v3"
  },
  "deepseek-r1-distill-llama-70b": {
    "Together": "deepseek-r1-distill-llama-70b",
    "PuterJS": [
      "openrouter:deepseek/deepseek-r1-distill-llama-70b:free",
      "openrouter:deepseek/deepseek-r1-distill-llama-70b"
    ],
    "Cerebras": "deepseek-r1-distill-llama-70b",
    "HuggingChat": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "OpenRouter": "deepseek/deepseek-r1-distill-llama-70b"
  },
  "deepseek-r1-distill-qwen-1.5b": {
    "Together": "deepseek-r1-distill-qwen-1.5b",
    "PuterJS": "openrouter:deepseek/deepseek-r1-distill-qwen-1.5b",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  },
  "deepseek-r1-distill-qwen-14b": {
    "Together": "deepseek-r1-distill-qwen-14b",
    "PuterJS": [
      "openrouter:deepseek/deepseek-r1-distill-qwen-14b:free",
      "openrouter:deepseek/deepseek-r1-distill-qwen-14b"
    ],
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  },
  "grok-2": {
    "Grok": "grok-2",
    "PuterJS": [
      "openrouter:x-ai/grok-2-vision-1212",
      "openrouter:x-ai/grok-2-1212"
    ]
  },
  "grok-3": {
    "Grok": "grok-3",
    "PuterJS": "x-ai:x-ai/grok-3",
    "ApiAirforce": "grok-3"
  },
  "grok-3-r1": {
    "Grok": "grok-3-reasoning"
  },
  "kimi-k2": {
    "HuggingFace": "moonshotai/Kimi-K2-Instruct",
    "Groq": "moonshotai/Kimi-K2-Instruct",
    "PuterJS": "openrouter:moonshotai/kimi-k2",
    "ApiAirforce": "kimi-k2",
    "HuggingChat": "moonshotai/Kimi-K2-Instruct",
    "OllamaSwarm": "kimi-k2:1t-cloud",
    "OpenRouter": "moonshotai/kimi-k2"
  },
  "sonar": {
    "PuterJS": "openrouter:perplexity/sonar",
    "OpenRouter": "perplexity/sonar",
    "PollinationsAI": "perplexity-fast"
  },
  "sonar-pro": {
    "PuterJS": "openrouter:perplexity/sonar-pro",
    "OpenRouter": "perplexity/sonar-pro",
    "PollinationsAI": "perplexity"
  },
  "sonar-reasoning": {
    "PuterJS": "openrouter:perplexity/sonar-reasoning",
    "PollinationsAI": "perplexity-reasoning"
  },
  "sonar-reasoning-pro": {
    "PuterJS": "openrouter:perplexity/sonar-reasoning-pro",
    "OpenRouter": "perplexity/sonar-reasoning-pro",
    "PollinationsAI": "perplexity-reasoning"
  },
  "r1-1776": {
    "Together": "r1-1776",
    "PuterJS": "openrouter:perplexity/r1-1776",
    "Perplexity": "r1-1776"
  },
  "miklium": {
    "Miklium": "miklium"
  },
  "surfsense": {
    "Surfsense": "surfsense"
  },
  "nemotron-70b": {
    "Together": "nemotron-70b",
    "HuggingChat": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "HuggingFace": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "PuterJS": "openrouter:nvidia/llama-3.1-nemotron-70b-instruct",
    "HuggingFaceAPI": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
  },
  "aria": {
    "OperaAria": "aria"
  },
  "sdxl-turbo": {
    "HuggingFaceMedia": "stabilityai/sdxl-turbo",
    "PollinationsImage": "sdxl-turbo",
    "HuggingFace": "stabilityai/sdxl-turbo",
    "HuggingChat": "stabilityai/sdxl-turbo",
    "PollinationsAI": "turbo"
  },
  "sd-3.5-large": {
    "HuggingFaceMedia": "stabilityai/stable-diffusion-3.5-large",
    "HuggingSpace": "stabilityai-stable-diffusion-3-5-large",
    "HuggingFace": "stabilityai/stable-diffusion-3.5-large",
    "HuggingChat": "stabilityai/stable-diffusion-3.5-large",
    "StabilityAI_SD35Large": "stabilityai-stable-diffusion-3-5-large"
  },
  "flux": {
    "HuggingFaceMedia": "black-forest-labs/FLUX.1-dev",
    "PollinationsImage": "flux",
    "Together": "flux",
    "HuggingSpace": "black-forest-labs-flux-1-dev",
    "HuggingFace": "black-forest-labs/FLUX.1-dev",
    "BlackForestLabs_Flux1Dev": "black-forest-labs-flux-1-dev",
    "HuggingChat": "black-forest-labs/FLUX.1-dev",
    "PollinationsAI": "flux"
  },
  "flux-pro": {
    "PollinationsImage": "flux-pro",
    "Together": "flux-pro",
    "PollinationsAI": "flux"
  },
  "flux-dev": {
    "PollinationsImage": "flux-dev",
    "HuggingSpace": "black-forest-labs-flux-1-dev",
    "Together": "flux-dev",
    "HuggingChat": "black-forest-labs/FLUX.1-dev",
    "HuggingFace": "black-forest-labs/FLUX.1-dev",
    "HuggingFaceMedia": "black-forest-labs/FLUX.1-dev",
    "BlackForestLabs_Flux1Dev": "black-forest-labs-flux-1-dev",
    "PollinationsAI": "flux"
  },
  "flux-schnell": {
    "PollinationsImage": "flux-schnell",
    "Together": "flux-schnell",
    "HuggingChat": "black-forest-labs/FLUX.1-schnell",
    "HuggingFace": "black-forest-labs/FLUX.1-schnell",
    "HuggingFaceMedia": "black-forest-labs/FLUX.1-schnell",
    "PollinationsAI": "flux"
  },
  "flux-redux": {
    "Together": "flux-redux"
  },
  "flux-depth": {
    "Together": "flux-depth"
  },
  "flux-canny": {
    "Together": "flux-canny"
  },
  "flux-kontext": {
    "PollinationsAI": "kontext",
    "Together": "flux-kontext"
  },
  "flux-dev-lora": {
    "Together": "flux-dev-lora"
  },
  "auto": {
    "OpenaiChat": "auto",
    "OpenRouter": "openrouter/auto"
  },
  "gpt-5.2": {
    "OpenaiChat": "gpt-5-2",
    "LMArena": "gpt-5.2",
    "PuterJS": "openai:openai/gpt-5.2",
    "ApiAirforce": "gpt-5.2",
    "OpenRouter": "openai/gpt-5.2",
    "PollinationsAI": "gpt-5.4"
  },
  "gpt-5.2-instant": {
    "OpenaiChat": "gpt-5-2-instant"
  },
  "gpt-5.2-thinking": {
    "OpenaiChat": "gpt-5-2-thinking"
  },
  "gpt-5.1": {
    "OpenaiChat": "gpt-5-1",
    "LMArena": "gpt-5.1",
    "PuterJS": "azure:openai/gpt-5.1",
    "ApiAirforce": "gpt-5.1",
    "OpenRouter": "openai/gpt-5.1"
  },
  "gpt-5.1-instant": {
    "OpenaiChat": "gpt-5-1-instant"
  },
  "gpt-5.1-thinking": {
    "OpenaiChat": "gpt-5-1-thinking"
  },
  "gpt-5": {
    "OpenaiChat": "gpt-5",
    "Copilot": "GPT-5",
    "Perplexity": "gpt-5",
    "EasyChat": "gpt-5-free",
    "PuterJS": "openai:openai/gpt-5",
    "ApiAirforce": "gpt-5",
    "CopilotApp": "smart",
    "CopilotSession": "GPT-5",
    "OpenRouter": "openai/gpt-5"
  },
  "gpt-5-instant": {
    "OpenaiChat": "gpt-5-instant"
  },
  "gpt-5-thinking": {
    "OpenaiChat": "gpt-5-thinking",
    "Perplexity": "gpt-5-thinking"
  },
  "chat": {
    "CopilotApp": "chat"
  },
  "openai": {
    "PollinationsAI": "openai"
  },
  "openai-fast": {
    "PollinationsAI": "openai-fast"
  },
  "gpt-5.4": {
    "PollinationsAI": "gpt-5.4",
    "PuterJS": "openai:openai/gpt-5.4",
    "ApiAirforce": "gpt-5.4",
    "OpenRouter": "openai/gpt-5.4"
  },
  "gpt-5.4-mini": {
    "PollinationsAI": "gpt-5.4-mini",
    "PuterJS": "azure:openai/gpt-5.4-mini",
    "ApiAirforce": "gpt-5.4-mini",
    "OpenRouter": "openai/gpt-5.4-mini"
  },
  "openai-large": {
    "PollinationsAI": "openai-large"
  },
  "mercury": {
    "PollinationsAI": "mercury",
    "LMArena": "mercury"
  },
  "qwen-coder": {
    "PollinationsAI": "qwen-coder",
    "OllamaSwarm": "qwen-coder:14b"
  },
  "mistral-small-3.2": {
    "PollinationsAI": "mistral-small-3.2"
  },
  "mistral": {
    "PollinationsAI": "mistral",
    "OllamaSwarm": "mistral:7b-instruct-v0.3-q4_0"
  },
  "openai-audio": {
    "PollinationsAI": "openai-audio"
  },
  "openai-audio-large": {
    "PollinationsAI": "openai-audio-large"
  },
  "gemini-3-flash": {
    "PollinationsAI": "gemini-3-flash",
    "LMArena": "gemini-3-flash",
    "PuterJS": "openrouter:google/gemini-3-flash-preview",
    "Antigravity": "gemini-3-flash",
    "ApiAirforce": "gemini-3-flash",
    "Ollama": "gemini-3-flash-preview",
    "OllamaSwarm": "gemini-3-flash-preview:cloud",
    "OpenRouter": "google/gemini-3-flash-preview"
  },
  "gemini": {
    "PollinationsAI": "gemini"
  },
  "gemini-flash-lite-3.1": {
    "PollinationsAI": "gemini-flash-lite-3.1"
  },
  "gemini-fast": {
    "PollinationsAI": "gemini-fast"
  },
  "deepseek": {
    "PollinationsAI": "deepseek",
    "ApiAirforce": "deepseek",
    "DeepSeekAPI": "deepseek-v3",
    "GlhfChat": "hf:deepseek-ai/DeepSeek-V3",
    "HuggingChat": "deepseek-ai/DeepSeek-V3",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-V3",
    "OllamaSwarm": "deepseek-v2:16b-lite-chat-q8_0",
    "PhindAi": "deepseek",
    "WeWordle": "v3"
  },
  "gemma": {
    "PollinationsAI": "gemma",
    "OllamaSwarm": "gemma:2b-instruct",
    "TeachAnything": "gemma"
  },
  "deepseek-pro": {
    "PollinationsAI": "deepseek-pro"
  },
  "grok": {
    "PollinationsAI": "grok",
    "Grok": "grok-latest",
    "PuterJS": [
      "openrouter:x-ai/grok-vision-beta",
      "openrouter:x-ai/grok-2-vision-1212",
      "openrouter:x-ai/grok-2-1212",
      "grok-beta",
      "grok-vision-beta",
      "openrouter:x-ai/grok-beta",
      "openrouter:x-ai/grok-3-beta",
      "openrouter:x-ai/grok-3-mini-beta"
    ]
  },
  "grok-4-20-reasoning": {
    "PollinationsAI": "grok-4-20-reasoning",
    "PuterJS": "x-ai:x-ai/grok-4-20-reasoning"
  },
  "grok-large": {
    "PollinationsAI": "grok-large"
  },
  "gemini-search": {
    "PollinationsAI": "gemini-search"
  },
  "gemini-search-fast": {
    "PollinationsAI": "gemini-search-fast"
  },
  "gemini-search-large": {
    "PollinationsAI": "gemini-search-large"
  },
  "midijourney": {
    "PollinationsAI": "midijourney"
  },
  "midijourney-large": {
    "PollinationsAI": "midijourney-large"
  },
  "claude-fast": {
    "PollinationsAI": "claude-fast"
  },
  "claude": {
    "PollinationsAI": "claude"
  },
  "claude-opus-4.6": {
    "PollinationsAI": "claude-opus-4.6",
    "ApiAirforce": "claude-opus-4.6",
    "OpenRouter": "anthropic/claude-opus-4.6"
  },
  "claude-opus-4.7": {
    "PollinationsAI": "claude-opus-4.7",
    "ApiAirforce": "claude-opus-4.7",
    "OpenRouter": "anthropic/claude-opus-4.7"
  },
  "claude-large": {
    "PollinationsAI": "claude-large"
  },
  "perplexity-fast": {
    "PollinationsAI": "perplexity-fast"
  },
  "perplexity-deep": {
    "PollinationsAI": "perplexity-deep"
  },
  "perplexity": {
    "PollinationsAI": "perplexity",
    "Perplexity": "perplexity"
  },
  "perplexity-reasoning": {
    "PollinationsAI": "perplexity-reasoning"
  },
  "kimi": {
    "PollinationsAI": "kimi",
    "PuterJS": "openrouter:~moonshotai/kimi-latest",
    "OpenRouter": "~moonshotai/kimi-latest"
  },
  "kimi-code": {
    "PollinationsAI": "kimi-code"
  },
  "gemini-large": {
    "PollinationsAI": "gemini-large"
  },
  "nova-fast": {
    "PollinationsAI": "nova-fast"
  },
  "nova": {
    "PollinationsAI": "nova",
    "OpenAIFM": "nova"
  },
  "glm": {
    "PollinationsAI": "glm"
  },
  "llama-": {
    "PollinationsAI": "llama"
  },
  "llama-maverick": {
    "PollinationsAI": "llama-maverick"
  },
  "llama-scout": {
    "PollinationsAI": "llama-scout"
  },
  "minimax-m2.7": {
    "PollinationsAI": "minimax-m2.7",
    "LMArena": "minimax-m2.7",
    "PuterJS": "minimax:minimax/minimax-m2.7",
    "ApiAirforce": "minimax-m2.7",
    "HuggingChat": "MiniMaxAI/MiniMax-M2.7",
    "HuggingFaceAPI": "MiniMaxAI/MiniMax-M2.7",
    "MiniMax": "MiniMax-M2.7",
    "Ollama": "minimax-m2.7",
    "OllamaSwarm": "minimax-m2.7:cloud",
    "OpenRouter": "minimax/minimax-m2.7"
  },
  "minimax": {
    "PollinationsAI": "minimax",
    "PuterJS": "openrouter:minimax/minimax-01",
    "HailuoAI": "minimax"
  },
  "mistral-large": {
    "PollinationsAI": "mistral-large",
    "OpenRouter": "mistralai/mistral-large"
  },
  "polly": {
    "PollinationsAI": "polly"
  },
  "qwen-coder-large": {
    "PollinationsAI": "qwen-coder-large"
  },
  "qwen-large": {
    "PollinationsAI": "qwen-large"
  },
  "qwen-vision": {
    "PollinationsAI": "qwen-vision"
  },
  "qwen-vision-pro": {
    "PollinationsAI": "qwen-vision-pro"
  },
  "step-flash": {
    "PollinationsAI": "step-flash"
  },
  "step-3.5-flash": {
    "PollinationsAI": "step-3.5-flash",
    "DeepInfra": "stepfun-ai/Step-3.5-Flash",
    "LMArena": "step-3.5-flash",
    "PuterJS": "openrouter:stepfun/step-3.5-flash",
    "ApiAirforce": "step-3.5-flash:free",
    "HuggingChat": "stepfun-ai/Step-3.5-Flash",
    "HuggingFaceAPI": "stepfun-ai/Step-3.5-Flash",
    "OpenRouter": "stepfun/step-3.5-flash"
  },
  "qwen-safety": {
    "PollinationsAI": "qwen-safety"
  },
  "sana": {
    "PollinationsAI": "sana"
  },
  "grok-4": {
    "Grok": "grok-4",
    "PuterJS": "x-ai:x-ai/grok-4",
    "ApiAirforce": "grok-4",
    "PollinationsAI": "grok"
  },
  "grok-4-heavy": {
    "Grok": "grok-4-heavy"
  },
  "grok-4-reasoning": {
    "Grok": "grok-4-reasoning"
  },
  "grok-3-reasoning": {
    "Grok": "grok-3-reasoning"
  },
  "grok-3-mini": {
    "Grok": "grok-3-mini",
    "PuterJS": "openrouter:x-ai/grok-3-mini-beta",
    "ApiAirforce": "grok-3-mini"
  },
  "grok-3-mini-reasoning": {
    "Grok": "grok-3-mini-reasoning"
  },
  "grok-2-image": {
    "Grok": "grok-2-image"
  },
  "qwen-3.7-plus": {
    "Qwen": "qwen3.7-plus",
    "LMArena": "qwen3.7-plus",
    "PuterJS": "togetherai:qwen/qwen3.7-plus",
    "OpenRouter": "qwen/qwen3.7-plus"
  },
  "qwen-3.7-max": {
    "Qwen": "qwen3.7-max",
    "LMArena": "qwen3.7-max",
    "PuterJS": "togetherai:qwen/qwen3.7-max",
    "OpenRouter": "qwen/qwen3.7-max"
  },
  "qwen-3.6-plus": {
    "Qwen": "qwen3.6-plus-preview",
    "LMArena": "qwen3.6-plus",
    "PuterJS": "alibaba:qwen/qwen3.6-plus",
    "OpenRouter": "qwen/qwen3.6-plus"
  },
  "qwen-3.6-max": {
    "Qwen": "qwen3.6-max-preview",
    "PuterJS": "alibaba:qwen/qwen3.6-max-preview",
    "OpenRouter": "qwen/qwen3.6-max-preview"
  },
  "qwen-3.6-27b": {
    "Qwen": "qwen3.6-27b",
    "LMArena": "qwen3.6-27b",
    "PuterJS": "alibaba:qwen/qwen3.6-27b",
    "HuggingChat": "Qwen/Qwen3.6-27B",
    "HuggingFaceAPI": "Qwen/Qwen3.6-27B",
    "OpenRouter": "qwen/qwen3.6-27b"
  },
  "qwen-series-invite-beta": {
    "Qwen": "qwen-latest-series-invite-beta-v16"
  },
  "qwen-3.5-plus": {
    "Qwen": "qwen3.5-plus",
    "PuterJS": "openrouter:qwen/qwen3.5-plus-20260420",
    "OpenRouter": "qwen/qwen3.5-plus-02-15"
  },
  "qwen-3.5-omni-plus": {
    "Qwen": "qwen3.5-omni-plus"
  },
  "qwen-3.6-35b-a3b": {
    "Qwen": "qwen3.6-35b-a3b",
    "DeepInfra": "Qwen/Qwen3.6-35B-A3B",
    "PuterJS": "alibaba:qwen/qwen3.6-35b-a3b",
    "ApiAirforce": "qwen3.6-35b-a3b",
    "HuggingChat": "Qwen/Qwen3.6-35B-A3B",
    "HuggingFaceAPI": "Qwen/Qwen3.6-35B-A3B",
    "OpenRouter": "qwen/qwen3.6-35b-a3b"
  },
  "qwen-3.5-flash": {
    "Qwen": "qwen3.5-flash",
    "LMArena": "qwen3.5-flash",
    "PuterJS": "openrouter:qwen/qwen3.5-flash-02-23",
    "OpenRouter": "qwen/qwen3.5-flash-02-23"
  },
  "qwen-3.5-max": {
    "Qwen": "qwen3.5-max-2026-03-08"
  },
  "qwen-3.5-397b-a17b": {
    "Qwen": "qwen3.5-397b-a17b",
    "DeepInfra": "Qwen/Qwen3.5-397B-A17B",
    "LMArena": "qwen3.5-397b-a17b",
    "PuterJS": "alibaba:qwen/qwen3.5-397b-a17b",
    "ApiAirforce": "qwen3.5-397b-a17b",
    "HuggingChat": "Qwen/Qwen3.5-397B-A17B",
    "HuggingFaceAPI": "Qwen/Qwen3.5-397B-A17B",
    "OpenRouter": "qwen/qwen3.5-397b-a17b"
  },
  "qwen-3.5-122b-a10b": {
    "Qwen": "qwen3.5-122b-a10b",
    "LMArena": "qwen3.5-122b-a10b",
    "PuterJS": "alibaba:qwen/qwen3.5-122b-a10b",
    "HuggingChat": "Qwen/Qwen3.5-122B-A10B",
    "HuggingFaceAPI": "Qwen/Qwen3.5-122B-A10B",
    "OpenRouter": "qwen/qwen3.5-122b-a10b"
  },
  "qwen-3.5-omni-flash": {
    "Qwen": "qwen3.5-omni-flash"
  },
  "qwen-3.5-27b": {
    "Qwen": "qwen3.5-27b",
    "LMArena": "qwen3.5-27b",
    "PuterJS": "alibaba:qwen/qwen3.5-27b",
    "HuggingChat": "Qwen/Qwen3.5-27B",
    "HuggingFaceAPI": "Qwen/Qwen3.5-27B",
    "OpenRouter": "qwen/qwen3.5-27b"
  },
  "qwen-3.5-35b-a3b": {
    "Qwen": "qwen3.5-35b-a3b",
    "LMArena": "qwen3.5-35b-a3b",
    "PuterJS": "alibaba:qwen/qwen3.5-35b-a3b",
    "HuggingChat": "Qwen/Qwen3.5-35B-A3B",
    "HuggingFaceAPI": "Qwen/Qwen3.5-35B-A3B",
    "OpenRouter": "qwen/qwen3.5-35b-a3b"
  },
  "qwen-3-max": {
    "Qwen": "qwen3-max-2026-01-23",
    "DeepInfra": "Qwen/Qwen3-Max",
    "LMArena": "qwen3-max-2025-09-26",
    "PuterJS": "alibaba:qwen/qwen3-max",
    "ApiAirforce": "qwen3-max",
    "OpenRouter": "qwen/qwen3-max"
  },
  "qwen-plus": {
    "Qwen": "qwen-plus-2025-07-28",
    "PuterJS": "openrouter:qwen/qwen-plus",
    "ApiAirforce": "qwen-plus",
    "OpenRouter": "qwen/qwen-plus"
  },
  "qwen-3-coder-plus": {
    "Qwen": "qwen3-coder-plus",
    "PuterJS": "alibaba:qwen/qwen3-coder-plus",
    "OpenRouter": "qwen/qwen3-coder-plus"
  },
  "qwen-3-vl-plus": {
    "Qwen": "qwen3-vl-plus",
    "PuterJS": "alibaba:qwen/qwen3-vl-plus"
  },
  "qwen-3-omni-flash": {
    "Qwen": "qwen3-omni-flash-2025-12-01",
    "LMArena": "qwen3-omni-flash",
    "PuterJS": "alibaba:qwen/qwen3-omni-flash"
  },
  "gpt-4o-mini-image": {
    "EasyChat": "gpt-4o-mini-image-free"
  },
  "grok-4.1-fast": {
    "EasyChat": "grok-4.1-fast-free",
    "ApiAirforce": "grok-4.1-fast"
  },
  "openrouter": {
    "EasyChat": "openrouter-free"
  },
  "glm-5.2": {
    "DeepInfra": "zai-org/GLM-5.2",
    "HuggingFace": "zai-org/GLM-5.2-FP8",
    "PuterJS": "z-ai:z-ai/glm-5.2",
    "ApiAirforce": "glm-5.2",
    "HuggingChat": "zai-org/GLM-5.2",
    "HuggingFaceAPI": "zai-org/GLM-5.2-FP8",
    "Ollama": "glm-5.2",
    "OllamaSwarm": "glm-5.2:cloud",
    "OpenRouter": "z-ai/glm-5.2",
    "PollinationsAI": "glm"
  },
  "kimi-k2.7-code": {
    "DeepInfra": "moonshotai/Kimi-K2.7-Code",
    "LMArena": "kimi-k2.7-code",
    "PuterJS": "togetherai:moonshotai/kimi-k2.7-code",
    "ApiAirforce": "kimi-k2.7-code",
    "HuggingChat": "moonshotai/Kimi-K2.7-Code",
    "HuggingFaceAPI": "moonshotai/Kimi-K2.7-Code",
    "Ollama": "kimi-k2.7-code",
    "OllamaSwarm": "kimi-k2.7-code:cloud",
    "OpenRouter": "moonshotai/kimi-k2.7-code",
    "PollinationsAI": "kimi-code"
  },
  "nvidia-nemotron-3-ultra-550b-a55b": {
    "DeepInfra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "HuggingChat": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "HuggingFaceAPI": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
  },
  "nemotron-3-nano-omni-30b-a3b-reasoning": {
    "DeepInfra": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
    "PuterJS": "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "OpenRouter": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
  },
  "deepseek-v4-flash": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Flash",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Flash",
    "LMArena": "deepseek-v4-flash",
    "PuterJS": "deepseek:deepseek/deepseek-v4-flash",
    "ApiAirforce": "deepseek-v4-flash",
    "HuggingChat": "deepseek-ai/DeepSeek-V4-Flash",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-V4-Flash",
    "Ollama": "deepseek-v4-flash",
    "OllamaSwarm": "deepseek-v4-flash:cloud",
    "OpenRouter": "deepseek/deepseek-v4-flash",
    "PollinationsAI": "deepseek"
  },
  "deepseek-v4-pro": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Pro",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Pro",
    "LMArena": "deepseek-v4-pro",
    "PuterJS": "deepseek:deepseek/deepseek-v4-pro",
    "ApiAirforce": "deepseek-v4-pro",
    "HuggingChat": "deepseek-ai/DeepSeek-V4-Pro",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-V4-Pro",
    "Ollama": "deepseek-v4-pro",
    "OllamaSwarm": "deepseek-v4-pro:cloud",
    "OpenRouter": "deepseek/deepseek-v4-pro",
    "PollinationsAI": "deepseek-pro"
  },
  "kimi-k2.6": {
    "DeepInfra": "moonshotai/Kimi-K2.6",
    "LMArena": "kimi-k2.6",
    "PuterJS": "moonshotai:moonshotai/kimi-k2.6",
    "ApiAirforce": "kimi-k2.6",
    "HuggingChat": "moonshotai/Kimi-K2.6",
    "HuggingFaceAPI": "moonshotai/Kimi-K2.6",
    "Ollama": "kimi-k2.6",
    "OllamaSwarm": "kimi-k2.6:cloud",
    "OpenRouter": "moonshotai/kimi-k2.6",
    "PollinationsAI": "kimi"
  },
  "mimo-v2.5": {
    "DeepInfra": "XiaomiMiMo/MiMo-V2.5",
    "LMArena": "mimo-v2.5",
    "PuterJS": "openrouter:xiaomi/mimo-v2.5",
    "ApiAirforce": "mimo-v2.5",
    "OpenRouter": "xiaomi/mimo-v2.5"
  },
  "mimo-v2.5-pro": {
    "DeepInfra": "XiaomiMiMo/MiMo-V2.5-Pro",
    "HuggingFace": "XiaomiMiMo/MiMo-V2.5-Pro",
    "LMArena": "mimo-v2.5-pro",
    "PuterJS": "openrouter:xiaomi/mimo-v2.5-pro",
    "ApiAirforce": "mimo-v2.5-pro",
    "HuggingFaceAPI": "XiaomiMiMo/MiMo-V2.5-Pro",
    "OpenRouter": "xiaomi/mimo-v2.5-pro"
  },
  "glm-5.1": {
    "DeepInfra": "zai-org/GLM-5.1",
    "HuggingFace": "zai-org/GLM-5.1",
    "LMArena": "glm-5.1",
    "PuterJS": "z-ai:z-ai/glm-5.1",
    "ApiAirforce": "glm-5.1",
    "HuggingChat": "zai-org/GLM-5.1-FP8",
    "HuggingFaceAPI": "zai-org/GLM-5.1-FP8",
    "Ollama": "glm-5.1",
    "OllamaSwarm": "glm-5.1:cloud",
    "OpenRouter": "z-ai/glm-5.1"
  },
  "gemma-4-26b-a4b-it": {
    "DeepInfra": "google/gemma-4-26B-A4B-it",
    "PuterJS": "openrouter:google/gemma-4-26b-a4b-it:free",
    "HuggingChat": "google/gemma-4-26B-A4B-it",
    "HuggingFaceAPI": "google/gemma-4-26B-A4B-it",
    "OpenRouter": "google/gemma-4-26b-a4b-it",
    "PollinationsAI": "gemma"
  },
  "gemma-4-31b-it": {
    "DeepInfra": "google/gemma-4-31B-it",
    "LMArena": "gemma-4-31b-it",
    "PuterJS": "togetherai:google/gemma-4-31b-it",
    "HuggingChat": "google/gemma-4-31B-it",
    "HuggingFaceAPI": "google/gemma-4-31B-it",
    "OpenRouter": "google/gemma-4-31b-it"
  },
  "nvidia-nemotron-3-super-120b-a12b": {
    "DeepInfra": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "HuggingFaceAPI": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
  },
  "glm-5": {
    "DeepInfra": "zai-org/GLM-5",
    "LMArena": "glm-5",
    "PuterJS": "z-ai:z-ai/glm-5",
    "ApiAirforce": "glm-5",
    "HuggingChat": "zai-org/GLM-5",
    "HuggingFaceAPI": "zai-org/GLM-5",
    "Ollama": "glm-5",
    "OllamaSwarm": "glm-5:cloud",
    "OpenRouter": "z-ai/glm-5"
  },
  "minimax-m2.5": {
    "DeepInfra": "MiniMaxAI/MiniMax-M2.5",
    "LMArena": "minimax-m2.5",
    "PuterJS": "minimax:minimax/minimax-m2.5",
    "ApiAirforce": "minimax-m2.5",
    "HuggingChat": "MiniMaxAI/MiniMax-M2.5",
    "Ollama": "minimax-m2.5",
    "OllamaSwarm": "minimax-m2.5:cloud",
    "OpenRouter": "minimax/minimax-m2.5",
    "PollinationsAI": "minimax-m2.7"
  },
  "qwen-3-max-thinking": {
    "DeepInfra": "Qwen/Qwen3-Max-Thinking",
    "LMArena": "qwen3-max-thinking",
    "PuterJS": "openrouter:qwen/qwen3-max-thinking",
    "OpenRouter": "qwen/qwen3-max-thinking"
  },
  "kimi-k2.5": {
    "DeepInfra": "moonshotai/Kimi-K2.5",
    "LMArena": "kimi-k2.5",
    "PuterJS": "moonshotai:moonshotai/kimi-k2.5",
    "ApiAirforce": "kimi-k2.5",
    "HuggingChat": "moonshotai/Kimi-K2.5",
    "HuggingFaceAPI": "moonshotai/Kimi-K2.5",
    "Ollama": "kimi-k2.5",
    "OllamaSwarm": "kimi-k2.5:cloud",
    "OpenRouter": "moonshotai/kimi-k2.5"
  },
  "glm-4.7-flash": {
    "DeepInfra": "zai-org/GLM-4.7-Flash",
    "PuterJS": "z-ai:z-ai/glm-4.7-flash",
    "ApiAirforce": "glm-4.7-flash",
    "HuggingChat": "zai-org/GLM-4.7-Flash",
    "HuggingFaceAPI": "zai-org/GLM-4.7-Flash",
    "OllamaSwarm": "glm-4.7-flash:latest",
    "OpenRouter": "z-ai/glm-4.7-flash"
  },
  "deepseek-v3.2": {
    "DeepInfra": "deepseek-ai/DeepSeek-V3.2",
    "PuterJS": "openrouter:deepseek/deepseek-v3.2",
    "ApiAirforce": "deepseek-v3.2",
    "HuggingChat": "deepseek-ai/DeepSeek-V3.2",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-V3.2",
    "Ollama": "deepseek-v3.2",
    "OllamaSwarm": "deepseek-v3.2:cloud",
    "OpenRouter": "deepseek/deepseek-v3.2"
  },
  "flux-2-klein-4b": {
    "DeepInfra": "black-forest-labs/FLUX-2-klein-4b"
  },
  "flux-2-klein-9b": {
    "DeepInfra": "black-forest-labs/FLUX-2-klein-9b"
  },
  "qwythos-9b-claude-mythos-5-1m": {
    "HuggingFace": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
    "HuggingFaceAPI": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"
  },
  "vibethinker-3b": {
    "HuggingFace": "WeiboAI/VibeThinker-3B",
    "HuggingFaceAPI": "WeiboAI/VibeThinker-3B"
  },
  "ornith-1.0-9b": {
    "HuggingFace": "deepreinforce-ai/Ornith-1.0-9B",
    "HuggingFaceAPI": "deepreinforce-ai/Ornith-1.0-9B"
  },
  "fastcontext-1.0-4b-sft": {
    "HuggingFace": "microsoft/FastContext-1.0-4B-SFT",
    "HuggingFaceAPI": "microsoft/FastContext-1.0-4B-SFT"
  },
  "qwable-9b-claude-fable-5": {
    "HuggingFace": "empero-ai/Qwable-9B-Claude-Fable-5",
    "HuggingFaceAPI": "empero-ai/Qwable-9B-Claude-Fable-5"
  },
  "qwable-5-27b-coder": {
    "HuggingFace": "DJLougen/Qwable-5-27B-Coder",
    "HuggingFaceAPI": "DJLougen/Qwable-5-27B-Coder"
  },
  "gemma-4-26b-a4b-styletune": {
    "HuggingFace": "Gryphe/Gemma-4-26B-A4B-StyleTune-V2",
    "HuggingFaceAPI": "Gryphe/Gemma-4-26B-A4B-StyleTune-V2"
  },
  "nvidia-nemotron-3-ultra-550b-a55b-nvfp4": {
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "HuggingChat": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "HuggingFaceAPI": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
  },
  "fastcontext-1.0-4b-rl": {
    "HuggingFace": "microsoft/FastContext-1.0-4B-RL",
    "HuggingFaceAPI": "microsoft/FastContext-1.0-4B-RL"
  },
  "qwen-3-8b": {
    "HuggingFace": "Qwen/Qwen3-8B",
    "PuterJS": [
      "openrouter:qwen/qwen3-8b:free",
      "openrouter:qwen/qwen3-8b"
    ],
    "ApiAirforce": "qwen3-8b",
    "HuggingChat": "Qwen/Qwen3-8B",
    "HuggingFaceAPI": "Qwen/Qwen3-8B",
    "OpenRouter": "qwen/qwen3-8b"
  },
  "qwen-3.6-27b-aeon-ultimate-uncensored": {
    "HuggingFace": "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16",
    "HuggingFaceAPI": "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16"
  },
  "qwen-3-coder-30b-a3b": {
    "HuggingFace": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "PuterJS": "alibaba:qwen/qwen3-coder-30b-a3b-instruct",
    "ApiAirforce": "qwen3-coder-30b-a3b",
    "HuggingChat": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "OpenRouter": "qwen/qwen3-coder-30b-a3b-instruct"
  },
  "qwen-3-coder-next": {
    "HuggingFace": "Qwen/Qwen3-Coder-Next",
    "PuterJS": "openrouter:qwen/qwen3-coder-next",
    "HuggingChat": "Qwen/Qwen3-Coder-Next",
    "HuggingFaceAPI": "Qwen/Qwen3-Coder-Next",
    "Ollama": "qwen3-coder-next",
    "OllamaSwarm": "qwen3-coder-next:q8_0",
    "OpenRouter": "qwen/qwen3-coder-next"
  },
  "llama-3.2-1b": {
    "HuggingFace": "meta-llama/Llama-3.2-1B-Instruct",
    "PuterJS": [
      "openrouter:meta-llama/llama-3.2-1b-instruct:free",
      "openrouter:meta-llama/llama-3.2-1b-instruct"
    ],
    "HuggingFaceAPI": "meta-llama/Llama-3.2-1B-Instruct",
    "OpenRouter": "meta-llama/llama-3.2-1b-instruct"
  },
  "z-image-engineer": {
    "HuggingFace": "BennyDaBall/Z-Image-Engineer-V6",
    "HuggingFaceAPI": "BennyDaBall/Z-Image-Engineer-V6"
  },
  "llama-3.2-11b-vision": {
    "HuggingFace": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "PuterJS": "openrouter:meta-llama/llama-3.2-11b-vision-instruct",
    "GlhfChat": "hf:meta-llama/Llama-3.2-11B-Vision-Instruct",
    "OpenRouter": "meta-llama/llama-3.2-11b-vision-instruct"
  },
  "command-r-plus24": {
    "HuggingFace": "CohereForAI/c4ai-command-r-plus-08-2024",
    "PuterJS": "openrouter:cohere/command-r-plus-08-2024",
    "HuggingSpace": "command-r-plus-08-2024",
    "CohereForAI_C4AI_Command": "command-r-plus-08-2024",
    "OpenRouter": "cohere/command-r-plus-08-2024"
  },
  "deepseek-r1-distill-qwen-32b": {
    "HuggingFace": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "PuterJS": [
      "openrouter:deepseek/deepseek-r1-distill-qwen-32b:free",
      "openrouter:deepseek/deepseek-r1-distill-qwen-32b"
    ],
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  },
  "llama-3.1-nemotron-70b": {
    "HuggingFace": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "GlhfChat": "hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
  },
  "mistral-nemo-2407": {
    "HuggingFace": "mistralai/Mistral-Nemo-Instruct-2407"
  },
  "krea-2-turbo": {
    "HuggingFace": "krea/Krea-2-Turbo",
    "HuggingFaceMedia": "krea/Krea-2-Turbo:fal-ai"
  },
  "krea-2-raw": {
    "HuggingFace": "krea/Krea-2-Raw"
  },
  "krea2.fp8": {
    "HuggingFace": "AlperKTS/Krea2_FP8"
  },
  "ideogram-4": {
    "HuggingFace": "ideogram-ai/ideogram-4-fp8",
    "HuggingFaceMedia": "ideogram-ai/ideogram-4-fp8:fal-ai"
  },
  "z-image-turbo": {
    "HuggingFace": "Tongyi-MAI/Z-Image-Turbo",
    "HuggingFaceMedia": "Tongyi-MAI/Z-Image-Turbo:wavespeed"
  },
  "krea-2-turbo-gguf": {
    "HuggingFace": "vantagewithai/Krea-2-Turbo-GGUF"
  },
  "ideogram.4.turbotime.lora": {
    "HuggingFace": "ostris/ideogram_4_turbotime_lora",
    "HuggingFaceMedia": "ostris/ideogram_4_turbotime_lora:fal-ai"
  },
  "krea-2-lora-retroanime": {
    "HuggingFace": "krea/Krea-2-LoRA-retroanime",
    "HuggingFaceMedia": "krea/Krea-2-LoRA-retroanime:fal-ai"
  },
  "ideogram-4-nf4": {
    "HuggingFace": "ideogram-ai/ideogram-4-nf4"
  },
  "qwen-image-edit-rapid-aio": {
    "HuggingFace": "Phr00t/Qwen-Image-Edit-Rapid-AIO"
  },
  "krea-2-realism-lora": {
    "HuggingFace": "gokaygokay/Krea-2-Realism-LoRA"
  },
  "flux2-klein-9b-uncensored-text-encoder": {
    "HuggingFace": "ponpoke/flux2-klein-9b-uncensored-text-encoder"
  },
  "llama-3": {
    "HuggingFace": "meta-llama/Llama-3.3-70B-Instruct",
    "HuggingChat": "meta-llama/Llama-3.3-70B-Instruct",
    "HuggingFaceAPI": "meta-llama/Llama-3.3-70B-Instruct",
    "OllamaSwarm": "llama3:8b-instruct-q4_K_M"
  },
  "moonshotai/Kimi-K2-Instruct": {
    "HuggingFace": "moonshotai/Kimi-K2-Instruct-0905",
    "HuggingChat": "moonshotai/Kimi-K2-Instruct-0905",
    "HuggingFaceAPI": "moonshotai/Kimi-K2-Instruct-0905"
  },
  "qvq-72b": {
    "HuggingFace": "Qwen/QVQ-72B-Preview",
    "HuggingChat": "Qwen/QVQ-72B-Preview",
    "HuggingFaceAPI": "Qwen/QVQ-72B-Preview"
  },
  "stable-diffusion-3.5-large": {
    "HuggingFace": "stabilityai/stable-diffusion-3.5-large",
    "HuggingFaceMedia": "stabilityai/stable-diffusion-3.5-large",
    "HuggingChat": "stabilityai/stable-diffusion-3.5-large"
  },
  "sdxl-1.0": {
    "HuggingFace": "stabilityai/stable-diffusion-xl-base-1.0",
    "HuggingFaceMedia": "stabilityai/stable-diffusion-xl-base-1.0",
    "HuggingChat": "stabilityai/stable-diffusion-xl-base-1.0"
  },
  "wan2.2-ti2v-5b": {
    "HuggingFaceMedia": "Wan-AI/Wan2.2-TI2V-5B:wavespeed"
  },
  "longcat-video": {
    "HuggingFaceMedia": "meituan-longcat/LongCat-Video:fal-ai"
  },
  "wan2.1-t2v-1.3b": {
    "HuggingFaceMedia": "Wan-AI/Wan2.1-T2V-1.3B:wavespeed"
  },
  "wan2.1-t2v-14b": {
    "HuggingFaceMedia": "Wan-AI/Wan2.1-T2V-14B:wavespeed"
  },
  "hunyuanvideo-1.5": {
    "HuggingFaceMedia": "tencent/HunyuanVideo-1.5:wavespeed"
  },
  "wan2.2-t2v-a14b": {
    "HuggingFaceMedia": "Wan-AI/Wan2.2-T2V-A14B:replicate"
  },
  "wan2.2-t2v-a14b-diffusers": {
    "HuggingFaceMedia": "Wan-AI/Wan2.2-T2V-A14B-Diffusers:replicate"
  },
  "cogvideox-5b": {
    "HuggingFaceMedia": "zai-org/CogVideoX-5b:fal-ai"
  },
  "mochi-1": {
    "HuggingFaceMedia": "genmo/mochi-1-preview:fal-ai"
  },
  "max": {
    "LMArena": "Max"
  },
  "claude-opus-4-6-thinking": {
    "LMArena": "claude-opus-4-6-thinking"
  },
  "claude-opus-4-7-thinking": {
    "LMArena": "claude-opus-4-7-thinking"
  },
  "claude-opus-4-6": {
    "LMArena": "claude-opus-4-6",
    "PuterJS": "anthropic:anthropic/claude-opus-4-6"
  },
  "claude-opus-4-7": {
    "LMArena": "claude-opus-4-7",
    "PuterJS": "anthropic:anthropic/claude-opus-4-7"
  },
  "gemini-3-pro": {
    "LMArena": "gemini-3-pro",
    "ApiAirforce": "gemini-3-pro"
  },
  "gpt-5.4-high-no-system-prompt": {
    "LMArena": "gpt-5.4-high-no-system-prompt"
  },
  "gpt-5.2-chat": {
    "LMArena": "gpt-5.2-chat-latest",
    "PuterJS": "openai:openai/gpt-5.2-chat",
    "ApiAirforce": "gpt-5.2-chat",
    "OpenRouter": "openai/gpt-5.2-chat"
  },
  "grok-4.20-beta-0309-reasoning": {
    "LMArena": "grok-4.20-beta-0309-reasoning"
  },
  "claude-opus-4-5-20251101-thinking-32k": {
    "LMArena": "claude-opus-4-5-20251101-thinking-32k"
  },
  "gpt-5.5-instant": {
    "LMArena": "gpt-5.5-instant"
  },
  "grok-4.20-multi-agent-beta-0309": {
    "LMArena": "grok-4.20-multi-agent-beta-0309"
  },
  "claude-sonnet-4-6": {
    "LMArena": "claude-sonnet-4-6",
    "PuterJS": "anthropic:anthropic/claude-sonnet-4-6"
  },
  "claude-opus-4-5": {
    "LMArena": "claude-opus-4-5-20251101",
    "PuterJS": "anthropic:anthropic/claude-opus-4-5"
  },
  "gpt-5.4-no-system-prompt": {
    "LMArena": "gpt-5.4-no-system-prompt"
  },
  "ernie-5.1": {
    "LMArena": "ernie-5.1-preview"
  },
  "kiteki": {
    "LMArena": "kiteki"
  },
  "kizen-beta": {
    "LMArena": "kizen-beta"
  },
  "claude-sonnet-4-5-20250929-thinking-32k": {
    "LMArena": "claude-sonnet-4-5-20250929-thinking-32k"
  },
  "claude-sonnet-4-5": {
    "LMArena": "claude-sonnet-4-5-20250929",
    "PuterJS": "anthropic:anthropic/claude-sonnet-4-5",
    "Antigravity": "claude-sonnet-4.5",
    "ApiAirforce": "claude-sonnet-4.5",
    "OpenRouter": "anthropic/claude-sonnet-4.5"
  },
  "dola-seed-2.0-pro-text": {
    "LMArena": "dola-seed-2.0-pro-text"
  },
  "gpt-5.1-high": {
    "LMArena": "gpt-5.1-high"
  },
  "claude-opus-4-1-20250805-thinking-16k": {
    "LMArena": "claude-opus-4-1-20250805-thinking-16k"
  },
  "gpt-5.3-chat": {
    "LMArena": "gpt-5.3-chat-latest",
    "PuterJS": "openrouter:openai/gpt-5.3-chat",
    "ApiAirforce": "gpt-5.3-chat",
    "OpenRouter": "openai/gpt-5.3-chat"
  },
  "mimo-v2-pro": {
    "LMArena": "mimo-v2-pro"
  },
  "minimax-m3": {
    "LMArena": "minimax-m3",
    "PuterJS": "minimax:minimax/minimax-m3",
    "ApiAirforce": "minimax-m3",
    "HuggingChat": "MiniMaxAI/MiniMax-M3",
    "HuggingFaceAPI": "MiniMaxAI/MiniMax-M3",
    "MiniMax": "MiniMax-M3",
    "Ollama": "minimax-m3",
    "OllamaSwarm": "minimax-m3:cloud",
    "OpenRouter": "minimax/minimax-m3",
    "PollinationsAI": "minimax"
  },
  "gpt-5.4-mini-high": {
    "LMArena": "gpt-5.4-mini-high"
  },
  "claude-opus-4-1": {
    "LMArena": "claude-opus-4-1-20250805",
    "PuterJS": "anthropic:anthropic/claude-opus-4-1",
    "Anthropic": "claude-opus-4-1-latest",
    "ApiAirforce": "claude-opus-4-1"
  },
  "gemini-2.5-pro-grounding-exp": {
    "LMArena": "gemini-2.5-pro-grounding-exp"
  },
  "glm-4.7": {
    "LMArena": "glm-4.7",
    "PuterJS": "z-ai:z-ai/glm-4.7",
    "ApiAirforce": "glm-4.7",
    "HuggingChat": "zai-org/GLM-4.7",
    "Ollama": "glm-4.7",
    "OllamaSwarm": "glm-4.7:cloud",
    "OpenRouter": "z-ai/glm-4.7"
  },
  "gpt-5.2-high": {
    "LMArena": "gpt-5.2-high"
  },
  "gpt-5-high": {
    "LMArena": "gpt-5-high"
  },
  "mimo-v2-omni": {
    "LMArena": "mimo-v2-omni"
  },
  "kimi-k2.5-instant": {
    "LMArena": "kimi-k2.5-instant"
  },
  "o3": {
    "LMArena": "o3-2025-04-16",
    "PuterJS": "openai:openai/o3",
    "ApiAirforce": "o3",
    "OpenRouter": "openai/o3"
  },
  "kimi-k2-thinking-turbo": {
    "LMArena": "kimi-k2-thinking-turbo"
  },
  "gpt-5-chat": {
    "LMArena": "gpt-5-chat",
    "PuterJS": "openai:openai/gpt-5-chat",
    "ApiAirforce": "gpt-5-chat",
    "OpenRouter": "openai/gpt-5-chat"
  },
  "claude-opus-4-20250514-thinking-16k": {
    "LMArena": "claude-opus-4-20250514-thinking-16k"
  },
  "qwen-3-235b-a22b-2507": {
    "LMArena": "qwen3-235b-a22b-instruct-2507",
    "PuterJS": "togetherai:qwen/qwen3-235b-a22b-instruct-2507-tput",
    "HuggingChat": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "HuggingFaceAPI": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "OpenRouter": "qwen/qwen3-235b-a22b-2507"
  },
  "kimi-k2-0905": {
    "LMArena": "kimi-k2-0905-preview",
    "PuterJS": "openrouter:moonshotai/kimi-k2-0905",
    "HuggingChat": "moonshotai/Kimi-K2-Instruct-0905",
    "HuggingFaceAPI": "moonshotai/Kimi-K2-Instruct-0905",
    "OpenRouter": "moonshotai/kimi-k2-0905"
  },
  "kimi-k2-0711": {
    "LMArena": "kimi-k2-0711-preview",
    "ApiAirforce": "kimi-k2-0711"
  },
  "deep-octo": {
    "LMArena": "deep-octo"
  },
  "mistral-large-3": {
    "LMArena": "mistral-large-3",
    "Ollama": "mistral-large-3:675b",
    "OllamaSwarm": "mistral-large-3:675b-cloud",
    "PollinationsAI": "mistral-large"
  },
  "qwen-3-vl-235b-a22b": {
    "LMArena": "qwen3-vl-235b-a22b-instruct",
    "PuterJS": "openrouter:qwen/qwen3-vl-235b-a22b-instruct",
    "ApiAirforce": "qwen3-vl-235b-a22b",
    "HuggingChat": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "OpenRouter": "qwen/qwen3-vl-235b-a22b-instruct"
  },
  "claude-opus-4": {
    "LMArena": "claude-opus-4-20250514",
    "PuterJS": "anthropic:anthropic/claude-opus-4",
    "Anthropic": "claude-opus-4-20250522",
    "ApiAirforce": "claude-opus-4",
    "OpenRouter": "anthropic/claude-opus-4"
  },
  "claude-haiku-4-5": {
    "LMArena": "claude-haiku-4-5-20251001",
    "PuterJS": "anthropic:anthropic/claude-haiku-4-5",
    "ApiAirforce": "claude-haiku-4.5",
    "OpenRouter": "anthropic/claude-haiku-4.5"
  },
  "mistral-medium-2508": {
    "LMArena": "mistral-medium-2508",
    "PuterJS": "mistralai:mistralai/mistral-medium-2508"
  },
  "qwen-3-235b-a22b-no-thinking": {
    "LMArena": "qwen3-235b-a22b-no-thinking"
  },
  "gpt-5.4-nano-high": {
    "LMArena": "gpt-5.4-nano-high"
  },
  "qwen-3-next-80b-a3b": {
    "LMArena": "qwen3-next-80b-a3b-instruct",
    "PuterJS": "openrouter:qwen/qwen3-next-80b-a3b-instruct:free",
    "ApiAirforce": "qwen3-next-80b-a3b",
    "HuggingChat": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "OpenRouter": "qwen/qwen3-next-80b-a3b-instruct"
  },
  "longcat-flash-chat": {
    "LMArena": "longcat-flash-chat"
  },
  "qwen-3-235b-a22b-thinking-2507": {
    "LMArena": "qwen3-235b-a22b-thinking-2507",
    "PuterJS": "openrouter:qwen/qwen3-235b-a22b-thinking-2507",
    "HuggingChat": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "OpenRouter": "qwen/qwen3-235b-a22b-thinking-2507"
  },
  "claude-sonnet-4-20250514-thinking-32k": {
    "LMArena": "claude-sonnet-4-20250514-thinking-32k"
  },
  "hunyuan-vision-1.5-thinking": {
    "LMArena": "hunyuan-vision-1.5-thinking"
  },
  "qwen-3-vl-235b-a22b-thinking": {
    "LMArena": "qwen3-vl-235b-a22b-thinking",
    "PuterJS": "openrouter:qwen/qwen3-vl-235b-a22b-thinking",
    "HuggingChat": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "OpenRouter": "qwen/qwen3-vl-235b-a22b-thinking"
  },
  "micro-mango": {
    "LMArena": "micro-mango"
  },
  "mimo-v2-flash (thinking)": {
    "LMArena": "mimo-v2-flash (thinking)"
  },
  "mimo-v2-flash": {
    "LMArena": "mimo-v2-flash"
  },
  "gpt-5-mini-high": {
    "LMArena": "gpt-5-mini-high"
  },
  "claude-sonnet-4": {
    "LMArena": "claude-sonnet-4-20250514",
    "PuterJS": "anthropic:anthropic/claude-sonnet-4",
    "Anthropic": "claude-sonnet-4-latest",
    "ApiAirforce": "claude-sonnet-4-20250514",
    "OpenRouter": "anthropic/claude-sonnet-4"
  },
  "qwen-3-coder-480b-a35b": {
    "LMArena": "qwen3-coder-480b-a35b-instruct",
    "PuterJS": "alibaba:qwen/qwen3-coder-480b-a35b-instruct",
    "ApiAirforce": "qwen3-coder-480b-a35b",
    "HuggingChat": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-Coder-480B-A35B-Instruct"
  },
  "mistral-medium-2505": {
    "LMArena": "mistral-medium-2505"
  },
  "minimax-m2.1": {
    "LMArena": "minimax-m2.1-preview",
    "PuterJS": "minimax:minimax/minimax-m2.1",
    "ApiAirforce": "minimax-m2.1",
    "HuggingChat": "MiniMaxAI/MiniMax-M2.1",
    "Ollama": "minimax-m2.1",
    "OllamaSwarm": "minimax-m2.1:cloud",
    "OpenRouter": "minimax/minimax-m2.1"
  },
  "qwen-3-30b-a3b-2507": {
    "LMArena": "qwen3-30b-a3b-instruct-2507",
    "PuterJS": "openrouter:qwen/qwen3-30b-a3b-instruct-2507",
    "HuggingFaceAPI": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "OpenRouter": "qwen/qwen3-30b-a3b-instruct-2507"
  },
  "trinity-large": {
    "LMArena": "trinity-large"
  },
  "qwen-3-235b-a22b": {
    "LMArena": "qwen3-235b-a22b",
    "PuterJS": "alibaba:qwen/qwen3-235b-a22b",
    "ApiAirforce": "qwen3-235b-a22b",
    "HuggingChat": "Qwen/Qwen3-235B-A22B",
    "HuggingFaceAPI": "Qwen/Qwen3-235B-A22B",
    "OpenRouter": "qwen/qwen3-235b-a22b"
  },
  "qwen-3-next-80b-a3b-thinking": {
    "LMArena": "qwen3-next-80b-a3b-thinking",
    "PuterJS": "alibaba:qwen/qwen3-next-80b-a3b-thinking",
    "HuggingChat": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "OpenRouter": "qwen/qwen3-next-80b-a3b-thinking"
  },
  "trinity-large-thinking": {
    "LMArena": "trinity-large-thinking",
    "PuterJS": "openrouter:arcee-ai/trinity-large-thinking",
    "HuggingFaceAPI": "arcee-ai/Trinity-Large-Thinking",
    "OpenRouter": "arcee-ai/trinity-large-thinking"
  },
  "gemma-3-27b-it": {
    "LMArena": "gemma-3-27b-it",
    "PuterJS": "openrouter:google/gemma-3-27b-it",
    "HuggingChat": "google/gemma-3-27b-it",
    "HuggingFaceAPI": "google/gemma-3-27b-it",
    "OpenRouter": "google/gemma-3-27b-it"
  },
  "minimax-m1": {
    "LMArena": "minimax-m1",
    "PuterJS": "openrouter:minimax/minimax-m1",
    "OpenRouter": "minimax/minimax-m1"
  },
  "grok-3-mini-high": {
    "LMArena": "grok-3-mini-high"
  },
  "march26-chatbot1": {
    "LMArena": "march26-chatbot1"
  },
  "mistral-small-2506": {
    "LMArena": "mistral-small-2506"
  },
  "grok-3-mini-beta": {
    "LMArena": "grok-3-mini-beta"
  },
  "intellect-3": {
    "LMArena": "intellect-3"
  },
  "mercury-2": {
    "LMArena": "mercury-2",
    "PuterJS": "openrouter:inception/mercury-2",
    "OpenRouter": "inception/mercury-2",
    "PollinationsAI": "mercury"
  },
  "ling-flash-2.0": {
    "LMArena": "ling-flash-2.0"
  },
  "minimax-m2": {
    "LMArena": "minimax-m2-preview",
    "PuterJS": "minimax:minimax/minimax-m2",
    "ApiAirforce": "minimax-m2",
    "HuggingChat": "MiniMaxAI/MiniMax-M2",
    "OllamaSwarm": "minimax-m2:cloud",
    "OpenRouter": "minimax/minimax-m2"
  },
  "nova-2-lite": {
    "LMArena": "nova-2-lite",
    "PuterJS": "openrouter:amazon/nova-2-lite-v1",
    "OpenRouter": "amazon/nova-2-lite-v1",
    "PollinationsAI": "nova"
  },
  "gpt-5-nano-high": {
    "LMArena": "gpt-5-nano-high"
  },
  "olmo-3.1-32b": {
    "LMArena": "olmo-3.1-32b-instruct"
  },
  "qwen-3-30b-a3b": {
    "LMArena": "qwen3-30b-a3b",
    "PuterJS": "openrouter:qwen/qwen3-30b-a3b",
    "ApiAirforce": "qwen3-30b-a3b",
    "OpenRouter": "qwen/qwen3-30b-a3b"
  },
  "ring-flash-2.0": {
    "LMArena": "ring-flash-2.0"
  },
  "gemma-3n-e4b-it": {
    "LMArena": "gemma-3n-e4b-it",
    "PuterJS": "togetherai:google/gemma-3n-e4b-it",
    "HuggingChat": "google/gemma-3n-E4B-it",
    "OpenRouter": "google/gemma-3n-e4b-it"
  },
  "gpt-oss-20b": {
    "LMArena": "gpt-oss-20b",
    "PuterJS": "togetherai:openai/gpt-oss-20b",
    "ApiAirforce": "gpt-oss-20b",
    "HuggingChat": "openai/gpt-oss-20b",
    "HuggingFaceAPI": "openai/gpt-oss-20b",
    "Ollama": "gpt-oss:20b",
    "OpenRouter": "openai/gpt-oss-20b"
  },
  "nvidia-nemotron-3-nano-30b-a3b": {
    "LMArena": "nvidia-nemotron-3-nano-30b-a3b-bf16",
    "HuggingFaceAPI": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  },
  "granite-4.1-8b": {
    "LMArena": "granite-4.1-8b",
    "PuterJS": "openrouter:ibm-granite/granite-4.1-8b",
    "OpenRouter": "ibm-granite/granite-4.1-8b"
  },
  "mistral-small-3.1-24b-2503": {
    "LMArena": "mistral-small-3.1-24b-instruct-2503",
    "HuggingFaceAPI": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  },
  "ibm-granite-h-small": {
    "LMArena": "ibm-granite-h-small"
  },
  "olmo-3.1-32b-think": {
    "LMArena": "olmo-3.1-32b-think"
  },
  "jebel.1": {
    "LMArena": "jebel_1"
  },
  "scooter": {
    "LMArena": "scooter"
  },
  "jebel.2": {
    "LMArena": "jebel_2"
  },
  "ring-2.5-1t": {
    "LMArena": "ring-2.5-1t"
  },
  "kira-star": {
    "LMArena": "kira-star"
  },
  "segesta": {
    "LMArena": "segesta"
  },
  "tetra-0429-1": {
    "LMArena": "tetra-0429-1"
  },
  "ember": {
    "LMArena": "ember"
  },
  "artemis.1": {
    "LMArena": "artemis_1"
  },
  "artemis.2": {
    "LMArena": "artemis_2"
  },
  "rover": {
    "LMArena": "rover"
  },
  "tetra-0429-2": {
    "LMArena": "tetra-0429-2"
  },
  "tetra-0429-3": {
    "LMArena": "tetra-0429-3"
  },
  "steed-0217": {
    "LMArena": "steed-0217"
  },
  "may26-chatbot3": {
    "LMArena": "may26-chatbot3"
  },
  "kiteki-beta": {
    "LMArena": "kiteki-beta"
  },
  "tetra-0429-4": {
    "LMArena": "tetra-0429-4"
  },
  "tetra-0429-5": {
    "LMArena": "tetra-0429-5"
  },
  "hofburg-1": {
    "LMArena": "hofburg-1"
  },
  "pulse": {
    "LMArena": "pulse"
  },
  "mivan": {
    "LMArena": "mivan"
  },
  "tetra-0429-6": {
    "LMArena": "tetra-0429-6"
  },
  "step-3.7-flash": {
    "LMArena": "step-3.7-flash",
    "PuterJS": "openrouter:stepfun/step-3.7-flash",
    "HuggingChat": "stepfun-ai/Step-3.7-Flash",
    "HuggingFaceAPI": "stepfun-ai/Step-3.7-Flash",
    "OpenRouter": "stepfun/step-3.7-flash",
    "PollinationsAI": "step-flash"
  },
  "maylynx-alpha": {
    "LMArena": "maylynx-alpha"
  },
  "cloud-buddy": {
    "LMArena": "cloud-buddy"
  },
  "luxor": {
    "LMArena": "luxor"
  },
  "pisces-0309": {
    "LMArena": "pisces-0309"
  },
  "pakson": {
    "LMArena": "pakson"
  },
  "may-beta": {
    "LMArena": "may-beta"
  },
  "ling-2.5-1t": {
    "LMArena": "ling-2.5-1t"
  },
  "globe.2": {
    "LMArena": "globe_2"
  },
  "dola-seed-2.0-preview-vision": {
    "LMArena": "dola-seed-2.0-preview-vision"
  },
  "mizar-beta": {
    "LMArena": "mizar-beta"
  },
  "beacon-qdn9": {
    "LMArena": "beacon-qdn9"
  },
  "anonymous-1800": {
    "LMArena": "anonymous-1800"
  },
  "gpt-5-high-no-system-prompt": {
    "LMArena": "gpt-5-high-no-system-prompt"
  },
  "spider": {
    "LMArena": "spider"
  },
  "mammoth-newt-0206": {
    "LMArena": "mammoth-newt-0206"
  },
  "pisces-0226d": {
    "LMArena": "pisces-0226d"
  },
  "hofburg.2": {
    "LMArena": "hofburg_2"
  },
  "solar-open2": {
    "LMArena": "solar-open2"
  },
  "pancetta": {
    "LMArena": "pancetta"
  },
  "victoria": {
    "LMArena": "victoria"
  },
  "pisces-0318-vision": {
    "LMArena": "pisces-0318-vision"
  },
  "myrion": {
    "LMArena": "myrion"
  },
  "march26-chatbot1-public": {
    "LMArena": "march26-chatbot1-public"
  },
  "june-alpha": {
    "LMArena": "june-alpha"
  },
  "tianyi": {
    "LMArena": "tianyi"
  },
  "maymo-beta": {
    "LMArena": "maymo-beta"
  },
  "maymo-alpha": {
    "LMArena": "maymo-alpha"
  },
  "pisces-0309-vision": {
    "LMArena": "pisces-0309-vision"
  },
  "hcx-lm-arena": {
    "LMArena": "hcx-lm-arena"
  },
  "blue-forge": {
    "LMArena": "blue-forge"
  },
  "anonymous-1111": {
    "LMArena": "anonymous-1111"
  },
  "rijks": {
    "LMArena": "rijks"
  },
  "spark": {
    "LMArena": "spark"
  },
  "pisces-0309b": {
    "LMArena": "pisces-0309b"
  },
  "hofburg.3": {
    "LMArena": "hofburg_3"
  },
  "leepwal": {
    "LMArena": "leepwal"
  },
  "blackhawk": {
    "LMArena": "blackhawk"
  },
  "hearth": {
    "LMArena": "hearth"
  },
  "happy-friday-testing-1": {
    "LMArena": "happy-friday-testing-1"
  },
  "scorch": {
    "LMArena": "scorch"
  },
  "beluga-0311-1": {
    "LMArena": "beluga-0311-1"
  },
  "monster": {
    "LMArena": "monster"
  },
  "dola-seed-2.0-pro-vision": {
    "LMArena": "dola-seed-2.0-pro-vision"
  },
  "monterey": {
    "LMArena": "monterey"
  },
  "february26-chatbot4": {
    "LMArena": "february26-chatbot4"
  },
  "whisperfall": {
    "LMArena": "whisperfall"
  },
  "neon": {
    "LMArena": "neon"
  },
  "anonymous-1835": {
    "LMArena": "anonymous-1835"
  },
  "mammoth-newt-0226": {
    "LMArena": "mammoth-newt-0226"
  },
  "viper": {
    "LMArena": "viper"
  },
  "anonymous-1825": {
    "LMArena": "anonymous-1825"
  },
  "nightride-on": {
    "LMArena": "nightride-on-v2"
  },
  "ring-1t": {
    "LMArena": "ring-1t",
    "HuggingFaceAPI": "inclusionAI/Ring-1T"
  },
  "zephyr": {
    "LMArena": "zephyr"
  },
  "pisces-0318-text": {
    "LMArena": "pisces-0318-text"
  },
  "pisces-0309c": {
    "LMArena": "pisces-0309c"
  },
  "redwood": {
    "LMArena": "redwood"
  },
  "yivon-beta": {
    "LMArena": "yivon-beta"
  },
  "atlas": {
    "LMArena": "atlas"
  },
  "vortex": {
    "LMArena": "vortex"
  },
  "tikal": {
    "LMArena": "tikal"
  },
  "bronze": {
    "LMArena": "bronze"
  },
  "march26-chatbot2": {
    "LMArena": "march26-chatbot2"
  },
  "march26-chatbot3": {
    "LMArena": "march26-chatbot3"
  },
  "karyu": {
    "LMArena": "karyu"
  },
  "botbot2": {
    "LMArena": "botbot2"
  },
  "orion": {
    "LMArena": "orion"
  },
  "pisces-0320": {
    "LMArena": "pisces-0320"
  },
  "duomo-1-hero": {
    "LMArena": "duomo-1-hero"
  },
  "anonymous-1218": {
    "LMArena": "anonymous-1218"
  },
  "zeylu-alpha": {
    "LMArena": "zeylu-alpha"
  },
  "zeylu-beta": {
    "LMArena": "zeylu-beta"
  },
  "clawl": {
    "LMArena": "clawl"
  },
  "ernie-exp-251024": {
    "LMArena": "ernie-exp-251024"
  },
  "minicpm-sala": {
    "LMArena": "minicpm-sala"
  },
  "anonymous-1221": {
    "LMArena": "anonymous-1221"
  },
  "u2": {
    "LMArena": "u2-preview"
  },
  "vierra": {
    "LMArena": "vierra"
  },
  "rotten-apple": {
    "LMArena": "rotten-apple"
  },
  "pisces-0309d": {
    "LMArena": "pisces-0309d"
  },
  "stephen": {
    "LMArena": "stephen-v2"
  },
  "yotta-nexus": {
    "LMArena": "yotta-nexus"
  },
  "queen-bee": {
    "LMArena": "queen-bee"
  },
  "eb45-vision": {
    "LMArena": "EB45-vision"
  },
  "pisces-llm-0130": {
    "LMArena": "pisces-llm-0130"
  },
  "february26-chatbot2": {
    "LMArena": "february26-chatbot2"
  },
  "february26-chatbot3": {
    "LMArena": "february26-chatbot3"
  },
  "clinkz": {
    "LMArena": "clinkz"
  },
  "velo": {
    "LMArena": "velo"
  },
  "dola-seed-2.0-preview-text": {
    "LMArena": "dola-seed-2.0-preview-text"
  },
  "kiwi-do": {
    "LMArena": "kiwi-do"
  },
  "eb45-turbo": {
    "LMArena": "EB45-turbo"
  },
  "raptor-1.8-0120": {
    "LMArena": "raptor-1.8-0120"
  },
  "mistral-medium-3.5": {
    "LMArena": "mistral-medium-3.5",
    "OllamaSwarm": "mistral-medium-3.5:latest"
  },
  "uros": {
    "LMArena": "uros"
  },
  "cosmic-clotho": {
    "LMArena": "cosmic-clotho"
  },
  "muse-spark": {
    "LMArena": "muse-spark"
  },
  "stephen-vision-csfix": {
    "LMArena": "stephen-vision-csfix"
  },
  "raptor-1123": {
    "LMArena": "raptor-1123"
  },
  "raptor-1124": {
    "LMArena": "raptor-1124"
  },
  "step-3-mini-2511": {
    "LMArena": "step-3-mini-2511"
  },
  "mimera": {
    "LMArena": "mimera"
  },
  "ernie-exp-vl-251016": {
    "LMArena": "ernie-exp-vl-251016"
  },
  "happy-friday-testing-2": {
    "LMArena": "happy-friday-testing-2"
  },
  "ling-1t": {
    "LMArena": "ling-1t"
  },
  "sunshine-ai": {
    "LMArena": "sunshine-ai"
  },
  "ernie-exp-251025": {
    "LMArena": "ernie-exp-251025"
  },
  "ernie-exp-251023": {
    "LMArena": "ernie-exp-251023"
  },
  "flying-octopus": {
    "LMArena": "flying-octopus"
  },
  "ernie-exp-251026": {
    "LMArena": "ernie-exp-251026"
  },
  "ernie-exp-251027": {
    "LMArena": "ernie-exp-251027"
  },
  "ling-1t-1031": {
    "LMArena": "ling-1t-1031"
  },
  "morian": {
    "LMArena": "morian"
  },
  "april26-chatbot1": {
    "LMArena": "april26-chatbot1"
  },
  "april26-chatbot2": {
    "LMArena": "april26-chatbot2"
  },
  "hofburg.4": {
    "LMArena": "hofburg_4"
  },
  "hofburg.5": {
    "LMArena": "hofburg_5"
  },
  "maelox": {
    "LMArena": "maelox"
  },
  "pteronura": {
    "LMArena": "pteronura"
  },
  "solar-eclipse": {
    "LMArena": "solar-eclipse"
  },
  "apex-atlas": {
    "LMArena": "apex-atlas"
  },
  "anonymous-1815": {
    "LMArena": "anonymous-1815"
  },
  "moryn": {
    "LMArena": "moryn"
  },
  "kizen-alpha": {
    "LMArena": "kizen-alpha"
  },
  "may26-chatbot4-public": {
    "LMArena": "may26-chatbot4-public"
  },
  "grok-4.20-beta1": {
    "LMArena": "grok-4.20-beta1"
  },
  "fiddle": {
    "LMArena": "fiddle"
  },
  "significant-otter": {
    "LMArena": "significant-otter"
  },
  "gpt-5.5-xhigh": {
    "LMArena": "gpt-5.5-xhigh"
  },
  "delphi": {
    "LMArena": "delphi"
  },
  "tetra-0505-1": {
    "LMArena": "tetra-0505-1"
  },
  "tetra-0505-2": {
    "LMArena": "tetra-0505-2"
  },
  "astral-lachesis": {
    "LMArena": "astral-lachesis"
  },
  "steed-0611": {
    "LMArena": "steed-0611"
  },
  "celestial-atropos": {
    "LMArena": "celestial-atropos"
  },
  "mistral-small-2603": {
    "LMArena": "mistral-small-2603",
    "PuterJS": "mistralai:mistralai/mistral-small-2603",
    "OpenRouter": "mistralai/mistral-small-2603",
    "PollinationsAI": "mistral"
  },
  "april26-chatbot3": {
    "LMArena": "april26-chatbot3"
  },
  "ernie-5.0-preview-1220": {
    "LMArena": "ernie-5.0-preview-1220"
  },
  "miyami": {
    "LMArena": "miyami"
  },
  "olympia": {
    "LMArena": "olympia"
  },
  "petra": {
    "LMArena": "petra"
  },
  "momoda-alpha": {
    "LMArena": "momoda-alpha"
  },
  "momoda-beta": {
    "LMArena": "momoda-beta"
  },
  "luxor-alt": {
    "LMArena": "luxor-alt"
  },
  "mymodel-v2-6-4wld": {
    "LMArena": "mymodel-v2-6-4wld"
  },
  "gpt-5.5-high": {
    "LMArena": "gpt-5.5-high"
  },
  "rc3-vscmodeld-s110-sgl": {
    "LMArena": "rc3-vscmodeld-s110-sgl"
  },
  "may26-chatbot4": {
    "LMArena": "may26-chatbot4"
  },
  "gpt-5-high-new-system-prompt": {
    "LMArena": "gpt-5-high-new-system-prompt"
  },
  "chickadee": {
    "LMArena": "chickadee"
  },
  "soren": {
    "LMArena": "soren"
  },
  "deepseek-v4-pro-thinking": {
    "LMArena": "deepseek-v4-pro-thinking"
  },
  "torin": {
    "LMArena": "torin"
  },
  "varga": {
    "LMArena": "varga"
  },
  "coolers": {
    "LMArena": "coolers"
  },
  "cooler": {
    "LMArena": "cooler"
  },
  "blankies": {
    "LMArena": "blankies"
  },
  "qwen-3-vl-8b-thinking": {
    "LMArena": "qwen3-vl-8b-thinking",
    "PuterJS": "openrouter:qwen/qwen3-vl-8b-thinking",
    "HuggingFaceAPI": "Qwen/Qwen3-VL-8B-Thinking",
    "OpenRouter": "qwen/qwen3-vl-8b-thinking"
  },
  "mekai": {
    "LMArena": "mekai"
  },
  "may26-chatbot1": {
    "LMArena": "may26-chatbot1"
  },
  "fusion": {
    "LMArena": "fusion",
    "PuterJS": "openrouter:openrouter/fusion",
    "OpenRouter": "openrouter/fusion"
  },
  "claude-fable-5": {
    "LMArena": "claude-fable-5",
    "PuterJS": "anthropic:anthropic/claude-fable-5",
    "OpenRouter": "anthropic/claude-fable-5"
  },
  "tetra-0507-1": {
    "LMArena": "tetra-0507-1"
  },
  "steed-0507": {
    "LMArena": "steed-0507"
  },
  "tetra-0507-4": {
    "LMArena": "tetra-0507-4"
  },
  "tetra-0507-2": {
    "LMArena": "tetra-0507-2"
  },
  "tetra-0507-5": {
    "LMArena": "tetra-0507-5"
  },
  "tetra-0507-3": {
    "LMArena": "tetra-0507-3"
  },
  "artemis.3": {
    "LMArena": "artemis_3"
  },
  "artemis.4": {
    "LMArena": "artemis_4"
  },
  "dark-matter": {
    "LMArena": "dark-matter"
  },
  "mylen": {
    "LMArena": "mylen"
  },
  "grok-4.3-high": {
    "LMArena": "grok-4.3-high"
  },
  "claude-opus-4-8-thinking": {
    "LMArena": "claude-opus-4-8-thinking"
  },
  "gemini-3-flash (thinking-minimal)": {
    "LMArena": "gemini-3-flash (thinking-minimal)"
  },
  "maelis": {
    "LMArena": "maelis"
  },
  "qwen-vl-max": {
    "LMArena": "qwen-vl-max-2025-08-13",
    "PuterJS": "openrouter:qwen/qwen-vl-max"
  },
  "may26-chatbot2": {
    "LMArena": "may26-chatbot2"
  },
  "qwen-3-vl-8b": {
    "LMArena": "qwen3-vl-8b-instruct",
    "PuterJS": "openrouter:qwen/qwen3-vl-8b-instruct",
    "ApiAirforce": "qwen3-vl-8b",
    "HuggingChat": "Qwen/Qwen3-VL-8B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-VL-8B-Instruct",
    "OpenRouter": "qwen/qwen3-vl-8b-instruct"
  },
  "gpt-5.5-dlp-test": {
    "LMArena": "gpt-5.5-dlp-test"
  },
  "grok-4.3": {
    "LMArena": "grok-4.3",
    "PuterJS": "azure:x-ai/grok-4.3",
    "ApiAirforce": "grok-4.3",
    "OpenRouter": "x-ai/grok-4.3",
    "PollinationsAI": "grok-large"
  },
  "may-alpha": {
    "LMArena": "may-alpha"
  },
  "deepseek-v4-flash-dlp-test": {
    "LMArena": "deepseek-v4-flash-dlp-test"
  },
  "glassy.lagoon": {
    "LMArena": "glassy_lagoon"
  },
  "emerald.lagoon": {
    "LMArena": "emerald_lagoon"
  },
  "gpt-5.5": {
    "LMArena": "gpt-5.5",
    "PuterJS": "openai:openai/gpt-5.5",
    "ApiAirforce": "gpt-5.5",
    "OpenRouter": "openai/gpt-5.5",
    "PollinationsAI": "openai-large"
  },
  "luna-hope": {
    "LMArena": "luna-hope"
  },
  "mira-sky": {
    "LMArena": "mira-sky"
  },
  "claude-opus-4-8": {
    "LMArena": "claude-opus-4-8",
    "PuterJS": "anthropic:anthropic/claude-opus-4-8"
  },
  "amazon.nova-pro": {
    "LMArena": "amazon.nova-pro-v1:0"
  },
  "deepseek-v4-flash-thinking": {
    "LMArena": "deepseek-v4-flash-thinking"
  },
  "melyora": {
    "LMArena": "melyora"
  },
  "glm-5v-turbo": {
    "LMArena": "glm-5v-turbo",
    "PuterJS": "z-ai:z-ai/glm-5v-turbo",
    "OpenRouter": "z-ai/glm-5v-turbo"
  },
  "gemini-3.1-flash-image-preview (nano-banana-2) [web-search]": {
    "LMArena": "gemini-3.1-flash-image-preview (nano-banana-2) [web-search]"
  },
  "gpt-image-1.5-high-fidelity": {
    "LMArena": "gpt-image-1.5-high-fidelity"
  },
  "gemini-3-pro-image-preview-2k (nano-banana-pro)": {
    "LMArena": "gemini-3-pro-image-preview-2k (nano-banana-pro)"
  },
  "nonnas-meatballs-open-weight": {
    "LMArena": "nonnas-meatballs-open-weight"
  },
  "recraft-v4.1-utility-pro": {
    "LMArena": "recraft-v4.1-utility-pro"
  },
  "flux-2-pro": {
    "LMArena": "flux-2-pro"
  },
  "left-bank": {
    "LMArena": "left-bank"
  },
  "flux-2-dev": {
    "LMArena": "flux-2-dev"
  },
  "seedream-4.5": {
    "LMArena": "seedream-4.5"
  },
  "seedream-5.0-lite": {
    "LMArena": "seedream-5.0-lite"
  },
  "recraft-v4.1-pro": {
    "LMArena": "recraft-v4.1-pro"
  },
  "imagen-4.0-generate": {
    "LMArena": "imagen-4.0-generate-001"
  },
  "qwen-image-2512": {
    "LMArena": "qwen-image-2512",
    "HuggingFaceMedia": "Qwen/Qwen-Image-2512:fal-ai"
  },
  "hidream-o1-image": {
    "LMArena": "hidream-o1-image"
  },
  "krea-2-medium": {
    "LMArena": "krea-2-medium"
  },
  "wan2.5": {
    "LMArena": "wan2.5-preview"
  },
  "wan2.5-t2i": {
    "LMArena": "wan2.5-t2i-preview"
  },
  "gpt-image-1-high-fidelity": {
    "LMArena": "gpt-image-1-high-fidelity"
  },
  "gpt-image-1": {
    "LMArena": "gpt-image-1"
  },
  "recraft": {
    "LMArena": "recraft-v3"
  },
  "wan2.7-image-pro": {
    "LMArena": "wan2.7-image-pro"
  },
  "krea-2-large": {
    "LMArena": "krea-2-large"
  },
  "wan2.7-image": {
    "LMArena": "wan2.7-image"
  },
  "seedream-3": {
    "LMArena": "seedream-3"
  },
  "z-image": {
    "LMArena": "z-image",
    "HuggingFaceMedia": "Tongyi-MAI/Z-Image:fal-ai"
  },
  "flux-1-kontext-max": {
    "LMArena": "flux-1-kontext-max"
  },
  "cosmos3-super": {
    "LMArena": "cosmos3-super"
  },
  "flux-1-kontext-pro": {
    "LMArena": "flux-1-kontext-pro"
  },
  "imagen-3.0-generate": {
    "LMArena": "imagen-3.0-generate-002"
  },
  "cosmos3-super-agentic": {
    "LMArena": "cosmos3-super-agentic"
  },
  "ideogram-v3-quality": {
    "LMArena": "ideogram-v3-quality"
  },
  "photon": {
    "LMArena": "photon"
  },
  "lucid-origin": {
    "LMArena": "lucid-origin"
  },
  "flux-1-kontext-dev": {
    "LMArena": "flux-1-kontext-dev"
  },
  "harbor": {
    "LMArena": "harbor"
  },
  "gpt-image-2 (medium)": {
    "LMArena": "gpt-image-2 (medium)"
  },
  "thunder": {
    "LMArena": "thunder"
  },
  "flow-state": {
    "LMArena": "flow-state"
  },
  "citrus": {
    "LMArena": "citrus"
  },
  "habanero": {
    "LMArena": "habanero"
  },
  "spinosaurus": {
    "LMArena": "spinosaurus"
  },
  "flow-state-2": {
    "LMArena": "flow-state-2"
  },
  "kelly": {
    "LMArena": "kelly"
  },
  "flow-state-3": {
    "LMArena": "flow-state-3"
  },
  "hidream-o1-image-1.5": {
    "LMArena": "hidream-o1-image-1.5"
  },
  "nonnos-meatballs-open-weight": {
    "LMArena": "nonnos-meatballs-open-weight"
  },
  "greenbean": {
    "LMArena": "greenbean"
  },
  "waffle": {
    "LMArena": "waffle"
  },
  "flashfennel": {
    "LMArena": "flashfennel"
  },
  "itadori-sv1": {
    "LMArena": "itadori-sv1"
  },
  "pebble-1": {
    "LMArena": "pebble-1"
  },
  "phantom.brush": {
    "LMArena": "phantom_brush"
  },
  "pebble-2": {
    "LMArena": "pebble-2"
  },
  "zen-bear": {
    "LMArena": "zen-bear-v4"
  },
  "auto-bear": {
    "LMArena": "auto-bear-v2"
  },
  "hidream-e1.1": {
    "LMArena": "hidream-e1.1"
  },
  "gcps-fast": {
    "LMArena": "gcps-fast"
  },
  "qwen-image-2.0": {
    "LMArena": "qwen-image-2.0"
  },
  "qwen-image-2.0-pro": {
    "LMArena": "qwen-image-2.0-pro"
  },
  "flashbrown-a": {
    "LMArena": "flashbrown-a"
  },
  "hunyuan-image-3.0-fal": {
    "LMArena": "hunyuan-image-3.0-fal"
  },
  "uni-1.1-max": {
    "LMArena": "uni-1.1-max"
  },
  "uni-1.1": {
    "LMArena": "uni-1.1"
  },
  "soft-shell": {
    "LMArena": "soft-shell"
  },
  "flashbrown-b": {
    "LMArena": "flashbrown-b"
  },
  "instant-ramen": {
    "LMArena": "instant-ramen"
  },
  "text-to-image-autoeval-test": {
    "LMArena": "text-to-image-autoeval-test"
  },
  "chives": {
    "LMArena": "chives"
  },
  "fennel": {
    "LMArena": "fennel"
  },
  "super-cara": {
    "LMArena": "super-cara"
  },
  "frenchfry": {
    "LMArena": "frenchfry"
  },
  "sungod": {
    "LMArena": "sungod"
  },
  "super-gcp": {
    "LMArena": "super-gcp"
  },
  "shakshouka": {
    "LMArena": "shakshouka"
  },
  "ellsworth": {
    "LMArena": "ellsworth"
  },
  "parasaurolophus": {
    "LMArena": "parasaurolophus"
  },
  "spectral.ink": {
    "LMArena": "spectral_ink"
  },
  "babylon": {
    "LMArena": "babylon"
  },
  "seededit-3.0": {
    "LMArena": "seededit-3.0"
  },
  "dialogue": {
    "LMArena": "dialogue"
  },
  "altair": {
    "LMArena": "altair"
  },
  "king-crab": {
    "LMArena": "king-crab"
  },
  "paper-lantern": {
    "LMArena": "paper-lantern"
  },
  "crepe": {
    "LMArena": "crepe"
  },
  "blue-crab": {
    "LMArena": "blue-crab"
  },
  "fennelbaby": {
    "LMArena": "fennelbaby"
  },
  "caudipteryx": {
    "LMArena": "caudipteryx"
  },
  "reve-v1.1-fast": {
    "LMArena": "reve-v1.1-fast"
  },
  "mussaurus": {
    "LMArena": "mussaurus"
  },
  "tyrannosaurus": {
    "LMArena": "tyrannosaurus"
  },
  "jalapeno": {
    "LMArena": "jalapeno"
  },
  "avalon": {
    "LMArena": "avalon"
  },
  "hotate": {
    "LMArena": "hotate"
  },
  "gemini-2.5-flash-image-preview (nano-banana)": {
    "LMArena": "gemini-2.5-flash-image-preview (nano-banana)"
  },
  "archaeopteryx": {
    "LMArena": "archaeopteryx"
  },
  "snow-crab": {
    "LMArena": "snow-crab"
  },
  "grok-imagine-image-quality": {
    "LMArena": "grok-imagine-image-quality"
  },
  "hunyuan-image-2.1": {
    "LMArena": "hunyuan-image-2.1"
  },
  "red-rock": {
    "LMArena": "red-rock"
  },
  "grok-imagine-image": {
    "LMArena": "grok-imagine-image",
    "ApiAirforce": "grok-imagine-image"
  },
  "dimetrodon": {
    "LMArena": "dimetrodon"
  },
  "phantom.quill": {
    "LMArena": "phantom_quill"
  },
  "qwen-image-edit": {
    "LMArena": "qwen-image-edit"
  },
  "wan2.5-i2i": {
    "LMArena": "wan2.5-i2i-preview"
  },
  "imagen-4.0-ultra-generate": {
    "LMArena": "imagen-4.0-ultra-generate-001"
  },
  "mondrian": {
    "LMArena": "mondrian"
  },
  "wan2.6-t2i": {
    "LMArena": "wan2.6-t2i"
  },
  "imagen-4.0-fast-generate": {
    "LMArena": "imagen-4.0-fast-generate-001"
  },
  "qwen-image-edit-2511": {
    "LMArena": "qwen-image-edit-2511"
  },
  "chatgpt-image-high-fidelity (20251216)": {
    "LMArena": "chatgpt-image-latest-high-fidelity (20251216)"
  },
  "wan2.6-image": {
    "LMArena": "wan2.6-image"
  },
  "qvq-max": {
    "PuterJS": "alibaba:qwen/qvq-max"
  },
  "qwen-flash": {
    "PuterJS": "alibaba:qwen/qwen-flash"
  },
  "qwen-max": {
    "PuterJS": "openrouter:qwen/qwen-max",
    "ApiAirforce": "qwen-max"
  },
  "qwen-mt-plus": {
    "PuterJS": "alibaba:qwen/qwen-mt-plus"
  },
  "qwen-mt-turbo": {
    "PuterJS": "alibaba:qwen/qwen-mt-turbo"
  },
  "qwen-omni-turbo": {
    "PuterJS": "alibaba:qwen/qwen-omni-turbo"
  },
  "qwen-turbo": {
    "PuterJS": "openrouter:qwen/qwen-turbo",
    "ApiAirforce": "qwen-turbo"
  },
  "qwen-vl-ocr": {
    "PuterJS": "alibaba:qwen/qwen-vl-ocr"
  },
  "qwen-vl-plus": {
    "PuterJS": "openrouter:qwen/qwen-vl-plus"
  },
  "qwen-2-5-14b": {
    "PuterJS": "alibaba:qwen/qwen2-5-14b-instruct"
  },
  "qwen-2-5-32b": {
    "PuterJS": "alibaba:qwen/qwen2-5-32b-instruct"
  },
  "qwen-2-5-72b": {
    "PuterJS": "alibaba:qwen/qwen2-5-72b-instruct"
  },
  "qwen-2-5-7b": {
    "PuterJS": "alibaba:qwen/qwen2-5-7b-instruct"
  },
  "qwen-2-5-omni-7b": {
    "PuterJS": "alibaba:qwen/qwen2-5-omni-7b"
  },
  "qwen-2-5-vl-72b": {
    "PuterJS": "alibaba:qwen/qwen2-5-vl-72b-instruct"
  },
  "qwen-2-5-vl-7b": {
    "PuterJS": "alibaba:qwen/qwen2-5-vl-7b-instruct"
  },
  "qwen-3-14b": {
    "PuterJS": [
      "openrouter:qwen/qwen3-14b:free",
      "openrouter:qwen/qwen3-14b"
    ],
    "ApiAirforce": "qwen3-14b",
    "HuggingChat": "Qwen/Qwen3-14B",
    "HuggingFaceAPI": "Qwen/Qwen3-14B",
    "OpenRouter": "qwen/qwen3-14b"
  },
  "qwen-3-coder-flash": {
    "PuterJS": "alibaba:qwen/qwen3-coder-flash",
    "OpenRouter": "qwen/qwen3-coder-flash"
  },
  "qwen-3-vl-30b-a3b": {
    "PuterJS": "openrouter:qwen/qwen3-vl-30b-a3b-instruct",
    "ApiAirforce": "qwen3-vl-30b-a3b",
    "HuggingChat": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "OpenRouter": "qwen/qwen3-vl-30b-a3b-instruct"
  },
  "qwq-plus": {
    "PuterJS": "alibaba:qwen/qwq-plus"
  },
  "claude-3-5-sonnet": {
    "PuterJS": "anthropic:anthropic/claude-3-5-sonnet-20240620",
    "Anthropic": "claude-3-5-sonnet-latest"
  },
  "claude-3-7-sonnet": {
    "PuterJS": "anthropic:anthropic/claude-3-7-sonnet",
    "Anthropic": "claude-3-7-sonnet-20250219"
  },
  "claude-3-haiku": {
    "PuterJS": [
      "claude-3-haiku-20240307",
      "openrouter:anthropic/claude-3-haiku:beta",
      "openrouter:anthropic/claude-3-haiku"
    ],
    "Anthropic": "claude-3-haiku-20240307",
    "OpenRouter": "anthropic/claude-3-haiku"
  },
  "gpt-5-codex": {
    "PuterJS": "azure:openai/gpt-5-codex",
    "ApiAirforce": "gpt-5-codex",
    "OpenRouter": "openai/gpt-5-codex"
  },
  "gpt-5-mini": {
    "PuterJS": "openai:openai/gpt-5-mini",
    "OpenRouter": "openai/gpt-5-mini",
    "PollinationsAI": "gpt-5.4-mini"
  },
  "gpt-5-nano": {
    "PuterJS": "openai:openai/gpt-5-nano",
    "ApiAirforce": "gpt-5-nano",
    "OpenRouter": "openai/gpt-5-nano",
    "PollinationsAI": "openai-fast"
  },
  "gpt-5.1-codex": {
    "PuterJS": "azure:openai/gpt-5.1-codex",
    "OpenRouter": "openai/gpt-5.1-codex"
  },
  "gpt-5.1-codex-mini": {
    "PuterJS": "azure:openai/gpt-5.1-codex-mini",
    "ApiAirforce": "gpt-5.1-codex-mini",
    "OpenRouter": "openai/gpt-5.1-codex-mini"
  },
  "gpt-5.2-codex": {
    "PuterJS": "azure:openai/gpt-5.2-codex",
    "ApiAirforce": "gpt-5.2-codex",
    "OpenRouter": "openai/gpt-5.2-codex"
  },
  "gpt-5.3-codex": {
    "PuterJS": "azure:openai/gpt-5.3-codex",
    "ApiAirforce": "gpt-5.3-codex",
    "OpenRouter": "openai/gpt-5.3-codex"
  },
  "gpt-5.4-nano": {
    "PuterJS": "azure:openai/gpt-5.4-nano",
    "ApiAirforce": "gpt-5.4-nano",
    "OpenRouter": "openai/gpt-5.4-nano",
    "PollinationsAI": "openai"
  },
  "grok-4-1-fast-non-reasoning": {
    "PuterJS": "azure:x-ai/grok-4-1-fast-non-reasoning",
    "PollinationsAI": "grok"
  },
  "grok-4-1-fast-reasoning": {
    "PuterJS": "azure:x-ai/grok-4-1-fast-reasoning",
    "PollinationsAI": "grok-4-20-reasoning"
  },
  "grok-4-20-non-reasoning": {
    "PuterJS": "x-ai:x-ai/grok-4-20-non-reasoning",
    "PollinationsAI": "grok"
  },
  "gemini-2.0-flash-lite": {
    "PuterJS": "google:google/gemini-2.0-flash-lite"
  },
  "gemini-2.5-flash-lite": {
    "PuterJS": "openrouter:google/gemini-2.5-flash-lite",
    "Antigravity": "gemini-2.5-flash-lite",
    "OpenRouter": "google/gemini-2.5-flash-lite",
    "PollinationsAI": "gemini-fast"
  },
  "minimax-m2.1-highspeed": {
    "PuterJS": "minimax:minimax/minimax-m2.1-highspeed"
  },
  "minimax-m2.5-highspeed": {
    "PuterJS": "minimax:minimax/minimax-m2.5-highspeed"
  },
  "minimax-m2.7-highspeed": {
    "PuterJS": "minimax:minimax/minimax-m2.7-highspeed",
    "MiniMax": "MiniMax-M2.7-highspeed"
  },
  "codestral-2508": {
    "PuterJS": "mistralai:mistralai/codestral-2508",
    "OpenRouter": "mistralai/codestral-2508"
  },
  "devstral-2512": {
    "PuterJS": "mistralai:mistralai/devstral-2512",
    "OpenRouter": "mistralai/devstral-2512"
  },
  "magistral-medium-2509": {
    "PuterJS": "mistralai:mistralai/magistral-medium-2509"
  },
  "magistral-small-2509": {
    "PuterJS": "mistralai:mistralai/magistral-small-2509"
  },
  "ministral-14b-2512": {
    "PuterJS": "mistralai:mistralai/ministral-14b-2512",
    "OpenRouter": "mistralai/ministral-14b-2512"
  },
  "ministral-3b-2512": {
    "PuterJS": "mistralai:mistralai/ministral-3b-2512",
    "OpenRouter": "mistralai/ministral-3b-2512"
  },
  "ministral-8b-2512": {
    "PuterJS": "mistralai:mistralai/ministral-8b-2512",
    "OpenRouter": "mistralai/ministral-8b-2512"
  },
  "mistral-large-2512": {
    "PuterJS": "mistralai:mistralai/mistral-large-2512",
    "OpenRouter": "mistralai/mistral-large-2512"
  },
  "mistral-medium-3-5": {
    "PuterJS": "mistralai:mistralai/mistral-medium-3-5",
    "OpenRouter": "mistralai/mistral-medium-3-5"
  },
  "open-mistral-nemo-2407": {
    "PuterJS": "mistralai:mistralai/open-mistral-nemo-2407"
  },
  "voxtral-small-2507": {
    "PuterJS": "mistralai:mistralai/voxtral-small-2507"
  },
  "moonshot-v1-128k": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-128k",
    "ApiAirforce": "moonshot-v1-128k"
  },
  "moonshot-v1-128k-vision": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-128k-vision-preview",
    "ApiAirforce": "moonshot-v1-128k-vision"
  },
  "moonshot-v1-32k": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-32k",
    "ApiAirforce": "moonshot-v1-32k"
  },
  "moonshot-v1-32k-vision": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-32k-vision-preview",
    "ApiAirforce": "moonshot-v1-32k-vision"
  },
  "moonshot-v1-8k": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-8k",
    "ApiAirforce": "moonshot-v1-8k"
  },
  "moonshot-v1-8k-vision": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-8k-vision-preview",
    "ApiAirforce": "moonshot-v1-8k-vision"
  },
  "moonshot-v1-auto": {
    "PuterJS": "moonshotai:moonshotai/moonshot-v1-auto"
  },
  "gpt-5.1-chat": {
    "PuterJS": "openai:openai/gpt-5.1-chat",
    "ApiAirforce": "gpt-5.1-chat",
    "OpenRouter": "openai/gpt-5.1-chat"
  },
  "gpt-5.2-pro": {
    "PuterJS": "openai:openai/gpt-5.2-pro",
    "OpenRouter": "openai/gpt-5.2-pro"
  },
  "gpt-5.4-pro": {
    "PuterJS": "openai:openai/gpt-5.4-pro",
    "ApiAirforce": "gpt-5.4-pro",
    "OpenRouter": "openai/gpt-5.4-pro"
  },
  "gpt-5.5-pro": {
    "PuterJS": "openai:openai/gpt-5.5-pro",
    "OpenRouter": "openai/gpt-5.5-pro"
  },
  "o1-pro": {
    "PuterJS": "openai:openai/o1-pro",
    "OpenRouter": "openai/o1-pro"
  },
  "o3-pro": {
    "PuterJS": "openai:openai/o3-pro",
    "OpenRouter": "openai/o3-pro"
  },
  "jamba-large-1.7": {
    "PuterJS": "openrouter:ai21/jamba-large-1.7",
    "OpenRouter": "ai21/jamba-large-1.7"
  },
  "aion-1.0": {
    "PuterJS": "openrouter:aion-labs/aion-1.0",
    "OpenRouter": "aion-labs/aion-1.0"
  },
  "aion-1.0-mini": {
    "PuterJS": "openrouter:aion-labs/aion-1.0-mini",
    "OpenRouter": "aion-labs/aion-1.0-mini"
  },
  "aion-2.0": {
    "PuterJS": "openrouter:aion-labs/aion-2.0",
    "OpenRouter": "aion-labs/aion-2.0"
  },
  "aion-rp-llama-3.1-8b": {
    "PuterJS": "openrouter:aion-labs/aion-rp-llama-3.1-8b",
    "OpenRouter": "aion-labs/aion-rp-llama-3.1-8b"
  },
  "olmo-3-32b-think": {
    "PuterJS": "openrouter:allenai/olmo-3-32b-think",
    "OpenRouter": "allenai/olmo-3-32b-think"
  },
  "nova-lite": {
    "PuterJS": "openrouter:amazon/nova-lite-v1",
    "OpenRouter": "amazon/nova-lite-v1"
  },
  "nova-micro": {
    "PuterJS": "openrouter:amazon/nova-micro-v1",
    "OpenRouter": "amazon/nova-micro-v1",
    "PollinationsAI": "nova-fast"
  },
  "nova-premier": {
    "PuterJS": "openrouter:amazon/nova-premier-v1",
    "OpenRouter": "amazon/nova-premier-v1"
  },
  "nova-pro": {
    "PuterJS": "openrouter:amazon/nova-pro-v1",
    "OpenRouter": "amazon/nova-pro-v1"
  },
  "magnum-v4-72b": {
    "PuterJS": "openrouter:anthracite-org/magnum-v4-72b",
    "OpenRouter": "anthracite-org/magnum-v4-72b"
  },
  "claude-opus-4.1": {
    "PuterJS": "openrouter:anthropic/claude-opus-4.1",
    "OpenRouter": "anthropic/claude-opus-4.1"
  },
  "claude-opus-4.6-fast": {
    "PuterJS": "openrouter:anthropic/claude-opus-4.6-fast",
    "OpenRouter": "anthropic/claude-opus-4.6-fast"
  },
  "claude-opus-4.7-fast": {
    "PuterJS": "openrouter:anthropic/claude-opus-4.7-fast",
    "OpenRouter": "anthropic/claude-opus-4.7-fast"
  },
  "claude-opus-4.8-fast": {
    "PuterJS": "openrouter:anthropic/claude-opus-4.8-fast",
    "OpenRouter": "anthropic/claude-opus-4.8-fast"
  },
  "coder-large": {
    "PuterJS": "openrouter:arcee-ai/coder-large",
    "OpenRouter": "arcee-ai/coder-large"
  },
  "trinity-mini": {
    "PuterJS": "openrouter:arcee-ai/trinity-mini",
    "OpenRouter": "arcee-ai/trinity-mini"
  },
  "virtuoso-large": {
    "PuterJS": "openrouter:arcee-ai/virtuoso-large",
    "OpenRouter": "arcee-ai/virtuoso-large"
  },
  "ernie-4.5-vl-424b-a47b": {
    "PuterJS": "openrouter:baidu/ernie-4.5-vl-424b-a47b",
    "OpenRouter": "baidu/ernie-4.5-vl-424b-a47b"
  },
  "seed-1.6": {
    "PuterJS": "openrouter:bytedance-seed/seed-1.6",
    "OpenRouter": "bytedance-seed/seed-1.6"
  },
  "seed-1.6-flash": {
    "PuterJS": "openrouter:bytedance-seed/seed-1.6-flash",
    "OpenRouter": "bytedance-seed/seed-1.6-flash"
  },
  "seed-2.0-lite": {
    "PuterJS": "openrouter:bytedance-seed/seed-2.0-lite",
    "OpenRouter": "bytedance-seed/seed-2.0-lite"
  },
  "seed-2.0-mini": {
    "PuterJS": "openrouter:bytedance-seed/seed-2.0-mini",
    "OpenRouter": "bytedance-seed/seed-2.0-mini"
  },
  "ui-tars-1.5-7b": {
    "PuterJS": "openrouter:bytedance/ui-tars-1.5-7b",
    "HuggingFaceAPI": "ByteDance-Seed/UI-TARS-1.5-7B",
    "OpenRouter": "bytedance/ui-tars-1.5-7b"
  },
  "dolphin-mistral-24b-venice-edition": {
    "PuterJS": "openrouter:cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "HuggingFaceAPI": "dphn/Dolphin-Mistral-24B-Venice-Edition",
    "OpenRouter": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
  },
  "command-r24": {
    "PuterJS": "openrouter:cohere/command-r-08-2024",
    "HuggingSpace": "command-r-08-2024",
    "CohereForAI_C4AI_Command": "command-r-08-2024",
    "HuggingChat": "CohereLabs/c4ai-command-r-08-2024",
    "OpenRouter": "cohere/command-r-08-2024"
  },
  "command-r7b24": {
    "PuterJS": "openrouter:cohere/command-r7b-12-2024",
    "HuggingSpace": "command-r7b-12-2024",
    "CohereForAI_C4AI_Command": "command-r7b-12-2024",
    "HuggingChat": "CohereLabs/c4ai-command-r7b-12-2024",
    "HuggingFaceAPI": "CohereLabs/c4ai-command-r7b-12-2024",
    "OpenRouter": "cohere/command-r7b-12-2024"
  },
  "north-mini-code": {
    "PuterJS": "openrouter:cohere/north-mini-code:free",
    "OpenRouter": "cohere/north-mini-code:free"
  },
  "cogito-v2.1-671b": {
    "PuterJS": "openrouter:deepcogito/cogito-v2.1-671b",
    "OpenRouter": "deepcogito/cogito-v2.1-671b"
  },
  "deepseek-chat-v3-0324": {
    "PuterJS": "openrouter:deepseek/deepseek-chat-v3-0324",
    "OpenRouter": "deepseek/deepseek-chat-v3-0324"
  },
  "deepseek-chat-v3.1": {
    "PuterJS": "openrouter:deepseek/deepseek-chat-v3.1",
    "OpenRouter": "deepseek/deepseek-chat-v3.1"
  },
  "deepseek-r1-0528": {
    "PuterJS": "openrouter:deepseek/deepseek-r1-0528",
    "HuggingChat": "deepseek-ai/DeepSeek-R1-0528",
    "OpenRouter": "deepseek/deepseek-r1-0528"
  },
  "deepseek-v3.1-terminus": {
    "PuterJS": "openrouter:deepseek/deepseek-v3.1-terminus",
    "ApiAirforce": "deepseek-v3.1-terminus",
    "HuggingChat": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "OpenRouter": "deepseek/deepseek-v3.1-terminus"
  },
  "deepseek-v3.2-exp": {
    "PuterJS": "openrouter:deepseek/deepseek-v3.2-exp",
    "HuggingChat": "deepseek-ai/DeepSeek-V3.2-Exp",
    "HuggingFaceAPI": "deepseek-ai/DeepSeek-V3.2-Exp",
    "OpenRouter": "deepseek/deepseek-v3.2-exp"
  },
  "gemini-2.5-flash-image": {
    "PuterJS": "openrouter:google/gemini-2.5-flash-image",
    "OpenRouter": "google/gemini-2.5-flash-image"
  },
  "gemini-2.5-flash-lite-preview25": {
    "PuterJS": "openrouter:google/gemini-2.5-flash-lite-preview-09-2025",
    "OpenRouter": "google/gemini-2.5-flash-lite-preview-09-2025"
  },
  "gemini-3-pro-image": {
    "PuterJS": "openrouter:google/gemini-3-pro-image-preview",
    "OpenRouter": "google/gemini-3-pro-image-preview"
  },
  "gemini-3.1-flash-image": {
    "PuterJS": "openrouter:google/gemini-3.1-flash-image-preview",
    "OpenRouter": "google/gemini-3.1-flash-image-preview"
  },
  "gemini-3.1-pro-preview-customtools": {
    "PuterJS": "openrouter:google/gemini-3.1-pro-preview-customtools",
    "ApiAirforce": "gemini-3.1-pro-preview-customtools",
    "OpenRouter": "google/gemini-3.1-pro-preview-customtools"
  },
  "gemma-2-27b-it": {
    "PuterJS": "openrouter:google/gemma-2-27b-it",
    "GlhfChat": "hf:google/gemma-2-27b-it",
    "OpenRouter": "google/gemma-2-27b-it"
  },
  "gemma-3-12b-it": {
    "PuterJS": "openrouter:google/gemma-3-12b-it",
    "HuggingChat": "google/gemma-3-12b-it",
    "HuggingFaceAPI": "google/gemma-3-12b-it",
    "OpenRouter": "google/gemma-3-12b-it"
  },
  "gemma-3-4b-it": {
    "PuterJS": "openrouter:google/gemma-3-4b-it",
    "HuggingChat": "google/gemma-3-4b-it",
    "HuggingFaceAPI": "google/gemma-3-4b-it",
    "OpenRouter": "google/gemma-3-4b-it"
  },
  "lyria-3-clip": {
    "PuterJS": "openrouter:google/lyria-3-clip-preview",
    "OpenRouter": "google/lyria-3-clip-preview"
  },
  "lyria-3-pro": {
    "PuterJS": "openrouter:google/lyria-3-pro-preview",
    "OpenRouter": "google/lyria-3-pro-preview"
  },
  "mythomax-l2-13b": {
    "PuterJS": "openrouter:gryphe/mythomax-l2-13b",
    "OpenRouter": "gryphe/mythomax-l2-13b"
  },
  "granite-4.0-h-micro": {
    "PuterJS": "openrouter:ibm-granite/granite-4.0-h-micro",
    "OpenRouter": "ibm-granite/granite-4.0-h-micro"
  },
  "ling-2.6-1t": {
    "PuterJS": "openrouter:inclusionai/ling-2.6-1t",
    "HuggingChat": "inclusionAI/Ling-2.6-1T",
    "OpenRouter": "inclusionai/ling-2.6-1t"
  },
  "ling-2.6-flash": {
    "PuterJS": "openrouter:inclusionai/ling-2.6-flash",
    "OpenRouter": "inclusionai/ling-2.6-flash"
  },
  "ring-2.6-1t": {
    "PuterJS": "openrouter:inclusionai/ring-2.6-1t",
    "OpenRouter": "inclusionai/ring-2.6-1t"
  },
  "inflection-3-pi": {
    "PuterJS": "openrouter:inflection/inflection-3-pi",
    "OpenRouter": "inflection/inflection-3-pi"
  },
  "inflection-3-productivity": {
    "PuterJS": "openrouter:inflection/inflection-3-productivity",
    "OpenRouter": "inflection/inflection-3-productivity"
  },
  "kat-coder-pro": {
    "PuterJS": "openrouter:kwaipilot/kat-coder-pro-v2",
    "OpenRouter": "kwaipilot/kat-coder-pro-v2"
  },
  "lfm-2-24b-a2b": {
    "PuterJS": "openrouter:liquid/lfm-2-24b-a2b",
    "OpenRouter": "liquid/lfm-2-24b-a2b"
  },
  "lfm-2.5-1.2b": {
    "PuterJS": "openrouter:liquid/lfm-2.5-1.2b-instruct:free",
    "OpenRouter": "liquid/lfm-2.5-1.2b-instruct:free"
  },
  "lfm-2.5-1.2b-thinking": {
    "PuterJS": "openrouter:liquid/lfm-2.5-1.2b-thinking:free",
    "OpenRouter": "liquid/lfm-2.5-1.2b-thinking:free"
  },
  "weaver": {
    "PuterJS": "openrouter:mancer/weaver",
    "OpenRouter": "mancer/weaver"
  },
  "phi-4": {
    "PuterJS": "openrouter:microsoft/phi-4",
    "HuggingFaceAPI": "microsoft/phi-4",
    "OllamaSwarm": "phi-4:14b",
    "OpenRouter": "microsoft/phi-4"
  },
  "phi-4-mini": {
    "PuterJS": "openrouter:microsoft/phi-4-mini-instruct",
    "HuggingFaceAPI": "microsoft/Phi-4-mini-instruct",
    "OpenRouter": "microsoft/phi-4-mini-instruct"
  },
  "wizardlm-2-8x22b": {
    "PuterJS": "openrouter:microsoft/wizardlm-2-8x22b",
    "HuggingChat": "alpindale/WizardLM-2-8x22B",
    "HuggingFaceAPI": "alpindale/WizardLM-2-8x22B",
    "OpenRouter": "microsoft/wizardlm-2-8x22b"
  },
  "minimax-01": {
    "PuterJS": "openrouter:minimax/minimax-01",
    "OpenRouter": "minimax/minimax-01"
  },
  "minimax-m2-her": {
    "PuterJS": "openrouter:minimax/minimax-m2-her",
    "OpenRouter": "minimax/minimax-m2-her"
  },
  "mistral-large-2407": {
    "PuterJS": "openrouter:mistralai/mistral-large-2407",
    "OpenRouter": "mistralai/mistral-large-2407"
  },
  "mistral-medium-3.1": {
    "PuterJS": "openrouter:mistralai/mistral-medium-3.1",
    "OpenRouter": "mistralai/mistral-medium-3.1"
  },
  "mistral-saba": {
    "PuterJS": "openrouter:mistralai/mistral-saba",
    "OpenRouter": "mistralai/mistral-saba"
  },
  "mistral-small-24b-2501": {
    "PuterJS": "openrouter:mistralai/mistral-small-24b-instruct-2501",
    "OpenRouter": "mistralai/mistral-small-24b-instruct-2501"
  },
  "mistral-small-3.2-24b": {
    "PuterJS": "openrouter:mistralai/mistral-small-3.2-24b-instruct",
    "OpenRouter": "mistralai/mistral-small-3.2-24b-instruct"
  },
  "mixtral-8x22b": {
    "PuterJS": "open-mixtral-8x22b",
    "OpenRouter": "mistralai/mixtral-8x22b-instruct"
  },
  "voxtral-small-24b-2507": {
    "PuterJS": "openrouter:mistralai/voxtral-small-24b-2507",
    "OpenRouter": "mistralai/voxtral-small-24b-2507"
  },
  "kimi-k2-thinking": {
    "PuterJS": "openrouter:moonshotai/kimi-k2-thinking",
    "HuggingChat": "moonshotai/Kimi-K2-Thinking",
    "HuggingFaceAPI": "moonshotai/Kimi-K2-Thinking",
    "OllamaSwarm": "kimi-k2-thinking:cloud",
    "OpenRouter": "moonshotai/kimi-k2-thinking"
  },
  "morph-v3-fast": {
    "PuterJS": "openrouter:morph/morph-v3-fast",
    "OpenRouter": "morph/morph-v3-fast"
  },
  "morph-v3-large": {
    "PuterJS": "openrouter:morph/morph-v3-large",
    "OpenRouter": "morph/morph-v3-large"
  },
  "nex-n2-pro": {
    "PuterJS": "openrouter:nex-agi/nex-n2-pro",
    "OpenRouter": "nex-agi/nex-n2-pro"
  },
  "hermes-3-llama-3.1-405b": {
    "PuterJS": "openrouter:nousresearch/hermes-3-llama-3.1-405b:free",
    "OpenRouter": "nousresearch/hermes-3-llama-3.1-405b"
  },
  "hermes-3-llama-3.1-70b": {
    "PuterJS": "openrouter:nousresearch/hermes-3-llama-3.1-70b",
    "OpenRouter": "nousresearch/hermes-3-llama-3.1-70b"
  },
  "hermes-4-405b": {
    "PuterJS": "openrouter:nousresearch/hermes-4-405b",
    "OpenRouter": "nousresearch/hermes-4-405b"
  },
  "hermes-4-70b": {
    "PuterJS": "openrouter:nousresearch/hermes-4-70b",
    "HuggingFaceAPI": "NousResearch/Hermes-4-70B",
    "OpenRouter": "nousresearch/hermes-4-70b"
  },
  "llama-3.3-nemotron-super-49b-v1.5": {
    "PuterJS": "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "OpenRouter": "nvidia/llama-3.3-nemotron-super-49b-v1.5"
  },
  "nemotron-3-nano-30b-a3b": {
    "PuterJS": "openrouter:nvidia/nemotron-3-nano-30b-a3b:free",
    "OpenRouter": "nvidia/nemotron-3-nano-30b-a3b"
  },
  "nemotron-3-super-120b-a12b": {
    "PuterJS": "openrouter:nvidia/nemotron-3-super-120b-a12b:free",
    "OpenRouter": "nvidia/nemotron-3-super-120b-a12b"
  },
  "nemotron-3-ultra-550b-a55b": {
    "PuterJS": "togetherai:nvidia/nemotron-3-ultra-550b-a55b",
    "OpenRouter": "nvidia/nemotron-3-ultra-550b-a55b"
  },
  "nemotron-3.5-content-safety": {
    "PuterJS": "openrouter:nvidia/nemotron-3.5-content-safety:free",
    "OpenRouter": "nvidia/nemotron-3.5-content-safety:free"
  },
  "nemotron-nano-12b-v2-vl": {
    "PuterJS": "openrouter:nvidia/nemotron-nano-12b-v2-vl:free",
    "OpenRouter": "nvidia/nemotron-nano-12b-v2-vl:free"
  },
  "nemotron-nano-9b": {
    "PuterJS": "openrouter:nvidia/nemotron-nano-9b-v2:free",
    "OpenRouter": "nvidia/nemotron-nano-9b-v2:free"
  },
  "gpt-3.5-turbo": {
    "PuterJS": "openrouter:openai/gpt-3.5-turbo-instruct",
    "OpenRouter": "openai/gpt-3.5-turbo"
  },
  "gpt-3.5-turbo-0613": {
    "PuterJS": "openrouter:openai/gpt-3.5-turbo-0613",
    "OpenRouter": "openai/gpt-3.5-turbo-0613"
  },
  "gpt-3.5-turbo-16k": {
    "PuterJS": "openrouter:openai/gpt-3.5-turbo-16k",
    "ApiAirforce": "gpt-3.5-turbo-16k",
    "OpenRouter": "openai/gpt-3.5-turbo-16k"
  },
  "gpt-4-turbo": {
    "PuterJS": "openrouter:openai/gpt-4-turbo-preview",
    "ApiAirforce": "gpt-4-turbo",
    "OpenRouter": "openai/gpt-4-turbo-preview"
  },
  "gpt-4o-mini-search": {
    "PuterJS": "openrouter:openai/gpt-4o-mini-search-preview",
    "ApiAirforce": "gpt-4o-mini-search",
    "OpenRouter": "openai/gpt-4o-mini-search-preview"
  },
  "gpt-4o-search": {
    "PuterJS": "openrouter:openai/gpt-4o-search-preview",
    "ApiAirforce": "gpt-4o-search",
    "OpenRouter": "openai/gpt-4o-search-preview"
  },
  "gpt-5-image": {
    "PuterJS": "openrouter:openai/gpt-5-image",
    "OpenRouter": "openai/gpt-5-image"
  },
  "gpt-5-image-mini": {
    "PuterJS": "openrouter:openai/gpt-5-image-mini",
    "OpenRouter": "openai/gpt-5-image-mini"
  },
  "gpt-5-pro": {
    "PuterJS": "openrouter:openai/gpt-5-pro",
    "OpenRouter": "openai/gpt-5-pro"
  },
  "gpt-5.1-codex-max": {
    "PuterJS": "openrouter:openai/gpt-5.1-codex-max",
    "OpenRouter": "openai/gpt-5.1-codex-max"
  },
  "gpt-5.4-image-2": {
    "PuterJS": "openrouter:openai/gpt-5.4-image-2",
    "OpenRouter": "openai/gpt-5.4-image-2"
  },
  "gpt-audio": {
    "PuterJS": "openrouter:openai/gpt-audio",
    "ApiAirforce": "gpt-audio",
    "OpenRouter": "openai/gpt-audio",
    "PollinationsAI": "openai-audio-large"
  },
  "gpt-audio-mini": {
    "PuterJS": "openrouter:openai/gpt-audio-mini",
    "OpenRouter": "openai/gpt-audio-mini",
    "PollinationsAI": "openai-audio"
  },
  "gpt-chat": {
    "PuterJS": "openrouter:openai/gpt-chat-latest",
    "OpenRouter": "openai/gpt-chat-latest"
  },
  "gpt-oss-safeguard-20b": {
    "PuterJS": "openrouter:openai/gpt-oss-safeguard-20b",
    "HuggingChat": "openai/gpt-oss-safeguard-20b",
    "HuggingFaceAPI": "openai/gpt-oss-safeguard-20b",
    "OpenRouter": "openai/gpt-oss-safeguard-20b"
  },
  "o3-deep-research": {
    "PuterJS": "openrouter:openai/o3-deep-research",
    "ApiAirforce": "o3-deep-research",
    "OpenRouter": "openai/o3-deep-research"
  },
  "o4-mini-deep-research": {
    "PuterJS": "openrouter:openai/o4-mini-deep-research",
    "ApiAirforce": "o4-mini-deep-research",
    "OpenRouter": "openai/o4-mini-deep-research"
  },
  "bodybuilder": {
    "PuterJS": "openrouter:openrouter/bodybuilder",
    "OpenRouter": "openrouter/bodybuilder"
  },
  "free": {
    "PuterJS": "openrouter:openrouter/free",
    "OpenRouter": "openrouter/free"
  },
  "owl-alpha": {
    "PuterJS": "openrouter:openrouter/owl-alpha",
    "OpenRouter": "openrouter/owl-alpha"
  },
  "pareto-code": {
    "PuterJS": "openrouter:openrouter/pareto-code",
    "OpenRouter": "openrouter/pareto-code"
  },
  "perceptron-mk1": {
    "PuterJS": "openrouter:perceptron/perceptron-mk1",
    "OpenRouter": "perceptron/perceptron-mk1"
  },
  "sonar-deep-research": {
    "PuterJS": "openrouter:perplexity/sonar-deep-research",
    "OpenRouter": "perplexity/sonar-deep-research"
  },
  "sonar-pro-search": {
    "PuterJS": "openrouter:perplexity/sonar-pro-search",
    "OpenRouter": "perplexity/sonar-pro-search"
  },
  "laguna-m.1": {
    "PuterJS": "openrouter:poolside/laguna-m.1:free",
    "OpenRouter": "poolside/laguna-m.1"
  },
  "laguna-xs.2": {
    "PuterJS": "openrouter:poolside/laguna-xs.2:free",
    "OpenRouter": "poolside/laguna-xs.2"
  },
  "qwen-3-30b-a3b-thinking-2507": {
    "PuterJS": "openrouter:qwen/qwen3-30b-a3b-thinking-2507",
    "OpenRouter": "qwen/qwen3-30b-a3b-thinking-2507"
  },
  "qwen-3-coder": {
    "PuterJS": "openrouter:qwen/qwen3-coder:free",
    "ApiAirforce": "qwen3-coder",
    "Ollama": "qwen3-coder:480b",
    "OllamaSwarm": "qwen3-coder:30b",
    "OpenRouter": "qwen/qwen3-coder"
  },
  "qwen-3-vl-30b-a3b-thinking": {
    "PuterJS": "openrouter:qwen/qwen3-vl-30b-a3b-thinking",
    "HuggingChat": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "OpenRouter": "qwen/qwen3-vl-30b-a3b-thinking"
  },
  "qwen-3-vl-32b": {
    "PuterJS": "openrouter:qwen/qwen3-vl-32b-instruct",
    "ApiAirforce": "qwen3-vl-32b",
    "OpenRouter": "qwen/qwen3-vl-32b-instruct"
  },
  "qwen-3.6-flash": {
    "PuterJS": "openrouter:qwen/qwen3.6-flash",
    "OpenRouter": "qwen/qwen3.6-flash"
  },
  "reka-edge": {
    "PuterJS": "openrouter:rekaai/reka-edge",
    "OpenRouter": "rekaai/reka-edge"
  },
  "reka-flash-3": {
    "PuterJS": "openrouter:rekaai/reka-flash-3",
    "OpenRouter": "rekaai/reka-flash-3"
  },
  "relace-apply-3": {
    "PuterJS": "openrouter:relace/relace-apply-3",
    "OpenRouter": "relace/relace-apply-3"
  },
  "relace-search": {
    "PuterJS": "openrouter:relace/relace-search",
    "OpenRouter": "relace/relace-search"
  },
  "fugu-ultra": {
    "PuterJS": "openrouter:sakana/fugu-ultra",
    "OpenRouter": "sakana/fugu-ultra"
  },
  "l3-lunaris-8b": {
    "PuterJS": "openrouter:sao10k/l3-lunaris-8b",
    "OpenRouter": "sao10k/l3-lunaris-8b"
  },
  "l3.1-70b-hanami-x1": {
    "PuterJS": "openrouter:sao10k/l3.1-70b-hanami-x1",
    "OpenRouter": "sao10k/l3.1-70b-hanami-x1"
  },
  "l3.1-euryale-70b": {
    "PuterJS": "openrouter:sao10k/l3.1-euryale-70b",
    "OpenRouter": "sao10k/l3.1-euryale-70b"
  },
  "l3.3-euryale-70b": {
    "PuterJS": "openrouter:sao10k/l3.3-euryale-70b",
    "OpenRouter": "sao10k/l3.3-euryale-70b"
  },
  "router": {
    "PuterJS": "openrouter:switchpoint/router",
    "OpenRouter": "switchpoint/router"
  },
  "hunyuan-a13b": {
    "PuterJS": "openrouter:tencent/hunyuan-a13b-instruct",
    "OpenRouter": "tencent/hunyuan-a13b-instruct"
  },
  "hy3": {
    "PuterJS": "openrouter:tencent/hy3-preview",
    "OpenRouter": "tencent/hy3-preview"
  },
  "cydonia-24b-v4.1": {
    "PuterJS": "openrouter:thedrummer/cydonia-24b-v4.1",
    "OpenRouter": "thedrummer/cydonia-24b-v4.1"
  },
  "rocinante-12b": {
    "PuterJS": "openrouter:thedrummer/rocinante-12b",
    "OpenRouter": "thedrummer/rocinante-12b"
  },
  "skyfall-36b": {
    "PuterJS": "openrouter:thedrummer/skyfall-36b-v2",
    "OpenRouter": "thedrummer/skyfall-36b-v2"
  },
  "unslopnemo-12b": {
    "PuterJS": "openrouter:thedrummer/unslopnemo-12b",
    "OpenRouter": "thedrummer/unslopnemo-12b"
  },
  "remm-slerp-l2-13b": {
    "PuterJS": "openrouter:undi95/remm-slerp-l2-13b",
    "OpenRouter": "undi95/remm-slerp-l2-13b"
  },
  "solar-pro-3": {
    "PuterJS": "openrouter:upstage/solar-pro-3",
    "OpenRouter": "upstage/solar-pro-3"
  },
  "palmyra-x5": {
    "PuterJS": "openrouter:writer/palmyra-x5",
    "OpenRouter": "writer/palmyra-x5"
  },
  "grok-4.20": {
    "PuterJS": "openrouter:x-ai/grok-4.20",
    "OpenRouter": "x-ai/grok-4.20"
  },
  "grok-4.20-multi-agent": {
    "PuterJS": "openrouter:x-ai/grok-4.20-multi-agent",
    "ApiAirforce": "grok-4.20-multi-agent",
    "OpenRouter": "x-ai/grok-4.20-multi-agent"
  },
  "grok-build-0.1": {
    "PuterJS": "openrouter:x-ai/grok-build-0.1",
    "OpenRouter": "x-ai/grok-build-0.1"
  },
  "gemini-flash": {
    "PuterJS": "openrouter:~google/gemini-flash-latest",
    "OpenRouter": "~google/gemini-flash-latest"
  },
  "gemini-pro": {
    "PuterJS": "openrouter:~google/gemini-pro-latest",
    "OpenRouter": "~google/gemini-pro-latest"
  },
  "gpt": {
    "PuterJS": "openrouter:~openai/gpt-latest",
    "OpenRouter": "~openai/gpt-latest"
  },
  "gpt-mini": {
    "PuterJS": "openrouter:~openai/gpt-mini-latest",
    "OpenRouter": "~openai/gpt-mini-latest"
  },
  "qwen-2-1.5b": {
    "PuterJS": "togetherai:arize-ai/qwen-2-1.5b-instruct"
  },
  "cogito-v2-1-671b": {
    "PuterJS": "togetherai:deepcogito/cogito-v2-1-671b"
  },
  "lfm2-24b-a2b": {
    "PuterJS": "togetherai:liquidai/lfm2-24b-a2b"
  },
  "llama-3.3-70b-turbo": {
    "PuterJS": "togetherai:meta-llama/llama-3.3-70b-instruct-turbo"
  },
  "llama-guard-4-12b": {
    "PuterJS": "togetherai:meta-llama/llama-guard-4-12b",
    "HuggingChat": "meta-llama/Llama-Guard-4-12B",
    "OpenRouter": "meta-llama/llama-guard-4-12b"
  },
  "llama-3-8b-lite": {
    "PuterJS": "togetherai:meta-llama/meta-llama-3-8b-instruct-lite"
  },
  "qwen-2.5-7b-turbo": {
    "PuterJS": "togetherai:qwen/qwen2.5-7b-instruct-turbo"
  },
  "qwen-3.5-9b": {
    "PuterJS": "togetherai:qwen/qwen3.5-9b",
    "HuggingChat": "Qwen/Qwen3.5-9B",
    "HuggingFaceAPI": "Qwen/Qwen3.5-9B",
    "OpenRouter": "qwen/qwen3.5-9b"
  },
  "grok-2-vision": {
    "PuterJS": "x-ai:x-ai/grok-2-vision",
    "ApiAirforce": "grok-2-vision"
  },
  "grok-2-vision-1212": {
    "PuterJS": "x-ai:x-ai/grok-2-vision-1212"
  },
  "grok-3-fast": {
    "PuterJS": "x-ai:x-ai/grok-3-fast",
    "ApiAirforce": "grok-3-fast"
  },
  "grok-3-mini-fast": {
    "PuterJS": "x-ai:x-ai/grok-3-mini-fast",
    "ApiAirforce": "grok-3-mini-fast"
  },
  "grok-4-0709": {
    "PuterJS": "x-ai:x-ai/grok-4-0709"
  },
  "grok-4-1-fast": {
    "PuterJS": "x-ai:x-ai/grok-4-1-fast",
    "PollinationsAI": "grok"
  },
  "grok-4-fast": {
    "PuterJS": "x-ai:x-ai/grok-4-fast",
    "ApiAirforce": "grok-4-fast",
    "PollinationsAI": "grok"
  },
  "grok-4-fast-non-reasoning": {
    "PuterJS": "x-ai:x-ai/grok-4-fast-non-reasoning"
  },
  "grok-beta": {
    "PuterJS": [
      "grok-beta",
      "grok-vision-beta",
      "openrouter:x-ai/grok-beta",
      "openrouter:x-ai/grok-3-beta"
    ]
  },
  "grok-code-fast-1": {
    "PuterJS": "x-ai:x-ai/grok-code-fast-1",
    "ApiAirforce": "grok-code-fast-1"
  },
  "grok-vision-beta": {
    "PuterJS": "x-ai:x-ai/grok-vision-beta"
  },
  "autoglm-phone-multilingual": {
    "PuterJS": "z-ai:z-ai/autoglm-phone-multilingual"
  },
  "glm-4-32b-0414-128k": {
    "PuterJS": "z-ai:z-ai/glm-4-32b-0414-128k"
  },
  "glm-4.5": {
    "PuterJS": "z-ai:z-ai/glm-4.5",
    "ApiAirforce": "glm-4.5",
    "HuggingChat": "zai-org/GLM-4.5",
    "HuggingFaceAPI": "zai-org/GLM-4.5",
    "OpenRouter": "z-ai/glm-4.5"
  },
  "glm-4.5-air": {
    "PuterJS": "z-ai:z-ai/glm-4.5-air",
    "ApiAirforce": "glm-4.5-air",
    "HuggingChat": "zai-org/GLM-4.5-Air",
    "OpenRouter": "z-ai/glm-4.5-air"
  },
  "glm-4.5-airx": {
    "PuterJS": "z-ai:z-ai/glm-4.5-airx"
  },
  "glm-4.5-flash": {
    "PuterJS": "z-ai:z-ai/glm-4.5-flash"
  },
  "glm-4.5-x": {
    "PuterJS": "z-ai:z-ai/glm-4.5-x"
  },
  "glm-4.5v": {
    "PuterJS": "z-ai:z-ai/glm-4.5v",
    "HuggingChat": "zai-org/GLM-4.5V-FP8",
    "HuggingFaceAPI": "zai-org/GLM-4.5V",
    "OpenRouter": "z-ai/glm-4.5v"
  },
  "glm-4.6": {
    "PuterJS": "z-ai:z-ai/glm-4.6",
    "ApiAirforce": "glm-4.6",
    "HuggingChat": "zai-org/GLM-4.6",
    "HuggingFaceAPI": "zai-org/GLM-4.6",
    "OllamaSwarm": "glm-4.6:cloud",
    "OpenRouter": "z-ai/glm-4.6"
  },
  "glm-4.6v": {
    "PuterJS": "z-ai:z-ai/glm-4.6v",
    "HuggingChat": "zai-org/GLM-4.6V-FP8",
    "OpenRouter": "z-ai/glm-4.6v"
  },
  "glm-4.6v-flash": {
    "PuterJS": "z-ai:z-ai/glm-4.6v-flash",
    "HuggingChat": "zai-org/GLM-4.6V-Flash"
  },
  "glm-4.6v-flashx": {
    "PuterJS": "z-ai:z-ai/glm-4.6v-flashx"
  },
  "glm-4.7-flashx": {
    "PuterJS": "z-ai:z-ai/glm-4.7-flashx"
  },
  "glm-5-turbo": {
    "PuterJS": "z-ai:z-ai/glm-5-turbo",
    "OpenRouter": "z-ai/glm-5-turbo"
  },
  "pixtral-large": {
    "PuterJS": "pixtral-large-latest"
  },
  "llama-3.3-8b": {
    "PuterJS": "openrouter:meta-llama/llama-3.3-8b-instruct:free"
  },
  "gemini-1.5-flash": {
    "PuterJS": [
      "gemini-1.5-flash",
      "openrouter:google/gemini-flash-1.5",
      "gemini-flash-1.5-8b"
    ]
  },
  "gemini-1.5-8b-flash": {
    "PuterJS": "openrouter:google/gemini-flash-1.5-8b"
  },
  "gemini-1.5-pro": {
    "PuterJS": "openrouter:google/gemini-pro-1.5"
  },
  "gemini-2.5-flash-thinking": {
    "PuterJS": "openrouter:google/gemini-2.5-flash-preview:thinking"
  },
  "gemma-2-9b": {
    "PuterJS": [
      "openrouter:google/gemma-2-9b-it:free",
      "openrouter:google/gemma-2-9b-it"
    ]
  },
  "gemma-3-1b": {
    "PuterJS": "openrouter:google/gemma-3-1b-it:free"
  },
  "gemma-3-4b": {
    "PuterJS": [
      "openrouter:google/gemma-3-4b-it:free",
      "openrouter:google/gemma-3-4b-it"
    ]
  },
  "gemma-3-12b": {
    "PuterJS": [
      "openrouter:google/gemma-3-12b-it:free",
      "openrouter:google/gemma-3-12b-it"
    ]
  },
  "hermes-2-pro": {
    "PuterJS": "openrouter:nousresearch/hermes-2-pro-llama-3-8b"
  },
  "hermes-3-70b": {
    "PuterJS": "openrouter:nousresearch/hermes-3-llama-3.1-70b"
  },
  "hermes-3-405b": {
    "PuterJS": "openrouter:nousresearch/hermes-3-llama-3.1-405b"
  },
  "deephermes-3-8b": {
    "PuterJS": "openrouter:nousresearch/deephermes-3-llama-3-8b-preview:free"
  },
  "deephermes-3-24b": {
    "PuterJS": "openrouter:nousresearch/deephermes-3-mistral-24b-preview:free"
  },
  "phi-3-mini": {
    "PuterJS": "openrouter:microsoft/phi-3-mini-128k-instruct"
  },
  "phi-3-medium": {
    "PuterJS": "openrouter:microsoft/phi-3-medium-128k-instruct"
  },
  "phi-4-multimodal": {
    "PuterJS": "openrouter:microsoft/phi-4-multimodal-instruct"
  },
  "phi-4-reasoning": {
    "PuterJS": "openrouter:microsoft/phi-4-reasoning:free"
  },
  "phi-4-reasoning-plus": {
    "PuterJS": [
      "openrouter:microsoft/phi-4-reasoning-plus:free",
      "openrouter:microsoft/phi-4-reasoning-plus"
    ]
  },
  "mai-ds-r1": {
    "PuterJS": "openrouter:microsoft/mai-ds-r1:free"
  },
  "claude-3.7-sonnet": {
    "PuterJS": [
      "claude-3-7-sonnet-20250219",
      "claude-3-7-sonnet-latest",
      "openrouter:anthropic/claude-3.7-sonnet",
      "openrouter:anthropic/claude-3.7-sonnet:beta"
    ]
  },
  "claude-3.7-sonnet-thinking": {
    "PuterJS": "openrouter:anthropic/claude-3.7-sonnet:thinking"
  },
  "claude-3.5-haiku": {
    "PuterJS": [
      "openrouter:anthropic/claude-3.5-haiku:beta",
      "openrouter:anthropic/claude-3.5-haiku",
      "openrouter:anthropic/claude-3.5-haiku-20241022:beta",
      "openrouter:anthropic/claude-3.5-haiku-20241022"
    ]
  },
  "claude-3.5-sonnet": {
    "PuterJS": [
      "claude-3-5-sonnet-20241022",
      "claude-3-5-sonnet-latest",
      "claude-3-5-sonnet-20240620",
      "openrouter:anthropic/claude-3.5-sonnet-20240620:beta",
      "openrouter:anthropic/claude-3.5-sonnet-20240620",
      "openrouter:anthropic/claude-3.5-sonnet:beta",
      "openrouter:anthropic/claude-3.5-sonnet"
    ]
  },
  "claude-3-opus": {
    "PuterJS": [
      "openrouter:anthropic/claude-3-opus:beta",
      "openrouter:anthropic/claude-3-opus"
    ],
    "Anthropic": "claude-3-opus-latest"
  },
  "claude-3-sonnet": {
    "PuterJS": [
      "openrouter:anthropic/claude-3-sonnet:beta",
      "openrouter:anthropic/claude-3-sonnet"
    ],
    "Anthropic": "claude-3-sonnet-20240229"
  },
  "claude-2.1": {
    "PuterJS": [
      "openrouter:anthropic/claude-2.1:beta",
      "openrouter:anthropic/claude-2.1"
    ]
  },
  "claude-2": {
    "PuterJS": [
      "openrouter:anthropic/claude-2:beta",
      "openrouter:anthropic/claude-2"
    ]
  },
  "claude-2.0": {
    "PuterJS": [
      "openrouter:anthropic/claude-2.0:beta",
      "openrouter:anthropic/claude-2.0"
    ]
  },
  "reka-flash": {
    "PuterJS": "openrouter:rekaai/reka-flash-3:free"
  },
  "command": {
    "PuterJS": "openrouter:cohere/command"
  },
  "qwen-2.5-vl-7b": {
    "PuterJS": [
      "openrouter:qwen/qwen-2.5-vl-7b-instruct:free",
      "openrouter:qwen/qwen-2.5-vl-7b-instruct"
    ]
  },
  "qwen-3-0.6b": {
    "PuterJS": "openrouter:qwen/qwen3-0.6b-04-28:free",
    "ApiAirforce": "qwen3-0.6b",
    "HuggingFaceAPI": "Qwen/Qwen3-0.6B"
  },
  "qwen-3-1.7b": {
    "PuterJS": "openrouter:qwen/qwen3-1.7b:free",
    "ApiAirforce": "qwen3-1.7b",
    "HuggingFaceAPI": "Qwen/Qwen3-1.7B"
  },
  "qwen-3-4b": {
    "PuterJS": "openrouter:qwen/qwen3-4b:free",
    "ApiAirforce": "qwen3-4b",
    "HuggingFaceAPI": "Qwen/Qwen3-4B"
  },
  "qwen-3-30b": {
    "PuterJS": [
      "openrouter:qwen/qwen3-30b-a3b:free",
      "openrouter:qwen/qwen3-30b-a3b"
    ]
  },
  "qwen-2.5-coder-7b": {
    "PuterJS": "openrouter:qwen/qwen2.5-coder-7b-instruct",
    "HuggingChat": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "HuggingFaceAPI": "Qwen/Qwen2.5-Coder-7B"
  },
  "qwen-2.5-vl-3b": {
    "PuterJS": "openrouter:qwen/qwen2.5-vl-3b-instruct:free"
  },
  "qwen-2.5-vl-32b": {
    "PuterJS": [
      "openrouter:qwen/qwen2.5-vl-32b-instruct:free",
      "openrouter:qwen/qwen2.5-vl-32b-instruct"
    ]
  },
  "deepseek-prover-v2": {
    "PuterJS": [
      "openrouter:deepseek/deepseek-prover-v2:free",
      "openrouter:deepseek/deepseek-prover-v2"
    ]
  },
  "deepseek-v3-0324": {
    "PuterJS": [
      "deepseek-chat",
      "openrouter:deepseek/deepseek-chat-v3-0324:free",
      "openrouter:deepseek/deepseek-chat-v3-0324"
    ],
    "ApiAirforce": "deepseek-v3-0324",
    "HuggingChat": "deepseek-ai/DeepSeek-V3-0324",
    "OllamaSwarm": "lordoliver/DeepSeek-V3-0324:671b-q4_k_m"
  },
  "deepseek-r1-zero": {
    "PuterJS": "openrouter:deepseek/deepseek-r1-zero:free"
  },
  "deepseek-r1-distill-llama-8b": {
    "PuterJS": "openrouter:deepseek/deepseek-r1-distill-llama-8b",
    "HuggingChat": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  },
  "deepseek-chat": {
    "PuterJS": [
      "deepseek-chat",
      "openrouter:deepseek/deepseek-chat:free",
      "openrouter:deepseek/deepseek-chat"
    ],
    "DeepSeekAPI": "deepseek-v3",
    "OpenRouter": "deepseek/deepseek-chat"
  },
  "deepseek-coder": {
    "PuterJS": [
      "openrouter:deepseek/deepseek-coder"
    ],
    "OllamaSwarm": "deepseek-coder:6.7b"
  },
  "grok-3-beta": {
    "PuterJS": "openrouter:x-ai/grok-3-beta"
  },
  "llama-3.1-sonar-small-online": {
    "PuterJS": "openrouter:perplexity/llama-3.1-sonar-small-128k-online"
  },
  "llama-3.1-sonar-large-online": {
    "PuterJS": "openrouter:perplexity/llama-3.1-sonar-large-128k-online"
  },
  "nemotron-49b": {
    "PuterJS": [
      "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1:free",
      "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1"
    ]
  },
  "nemotron-253b": {
    "PuterJS": "openrouter:nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
  },
  "glm-4": {
    "PuterJS": [
      "openrouter:thudm/glm-4-32b:free",
      "openrouter:thudm/glm-4-32b",
      "openrouter:thudm/glm-4-9b:free"
    ],
    "ApiAirforce": "glm-4"
  },
  "glm-4-32b": {
    "PuterJS": [
      "openrouter:thudm/glm-4-32b:free",
      "openrouter:thudm/glm-4-32b"
    ]
  },
  "glm-z1-32b": {
    "PuterJS": [
      "openrouter:thudm/glm-z1-32b:free",
      "openrouter:thudm/glm-z1-32b"
    ]
  },
  "glm-4-9b": {
    "PuterJS": "openrouter:thudm/glm-4-9b:free"
  },
  "glm-z1-9b": {
    "PuterJS": "openrouter:thudm/glm-z1-9b:free"
  },
  "glm-z1-rumination-32b": {
    "PuterJS": "openrouter:thudm/glm-z1-rumination-32b"
  },
  "dolphin-3.0-r1-24b": {
    "PuterJS": "openrouter:cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
  },
  "dolphin-3.0-24b": {
    "PuterJS": "openrouter:cognitivecomputations/dolphin3.0-mistral-24b:free"
  },
  "dolphin-8x22b": {
    "PuterJS": "openrouter:cognitivecomputations/dolphin-mixtral-8x22b"
  },
  "deepcoder-14b": {
    "PuterJS": "openrouter:agentica-org/deepcoder-14b-preview:free"
  },
  "kimi-vl-thinking": {
    "PuterJS": "openrouter:moonshotai/kimi-vl-a3b-thinking:free"
  },
  "moonlight-16b": {
    "PuterJS": "openrouter:moonshotai/moonlight-16b-a3b-instruct:free"
  },
  "qwerky-72b": {
    "PuterJS": "openrouter:featherless/qwerky-72b:free"
  },
  "lfm-7b": {
    "PuterJS": "openrouter:liquid/lfm-7b"
  },
  "lfm-3b": {
    "PuterJS": "openrouter:liquid/lfm-3b"
  },
  "lfm-40b": {
    "PuterJS": "openrouter:liquid/lfm-40b"
  },
  "command-a25": {
    "HuggingSpace": "command-a-03-2025",
    "CohereForAI_C4AI_Command": "command-a-03-2025",
    "HuggingChat": "CohereLabs/c4ai-command-a-03-2025"
  },
  "command-r7b-arabic25": {
    "HuggingSpace": "command-r7b-arabic-02-2025",
    "CohereForAI_C4AI_Command": "command-r7b-arabic-02-2025",
    "HuggingChat": "CohereLabs/c4ai-command-r7b-arabic-02-2025"
  },
  "flux-kontext-dev": {
    "HuggingSpace": "flux-kontext-dev",
    "BlackForestLabs_Flux1KontextDev": "flux-kontext-dev"
  },
  "pplx_pro": {
    "Perplexity": "pplx_pro"
  },
  "video": {
    "Video": "video"
  }
}
models_count = {
  "default": 11,
  "gpt-4": 10,
  "gpt-4o": 8,
  "gpt-4o-mini": 5,
  "gpt-4o-mini-tts": 2,
  "o1": 9,
  "o1-mini": 3,
  "o3-mini": 6,
  "o3-mini-high": 4,
  "o4-mini": 5,
  "o4-mini-high": 3,
  "gpt-4.1": 5,
  "gpt-4.1-mini": 4,
  "gpt-4.1-nano": 4,
  "gpt-4.5": 2,
  "gpt-oss-120b": 10,
  "smart": 3,
  "reasoning": 2,
  "study": 4,
  "search": 3,
  "dall-e-3": 6,
  "gpt-image": 3,
  "meta-ai": 2,
  "llama-2-70b": 2,
  "llama-3-8b": 6,
  "llama-3-70b": 5,
  "llama-3.1-8b": 8,
  "llama-3.1-70b": 6,
  "llama-3.1-405b": 3,
  "llama-3.2-3b": 6,
  "llama-3.2-11b": 5,
  "llama-3.2-90b": 2,
  "llama-3.3-70b": 9,
  "llama-4-scout": 4,
  "llama-4-maverick": 4,
  "mistral-nemo": 5,
  "mistral-small-3.1-24b": 3,
  "hermes-2-dpo": 2,
  "phi-3.5-mini": 4,
  "gemini-2.0-flash": 4,
  "gemini-2.0-flash-thinking": 2,
  "gemini-2.5-flash": 8,
  "gemini-2.5-pro": 9,
  "gemini-3.1-pro": 6,
  "gemini-3.1-flash-lite": 6,
  "gemini-3.5-flash": 6,
  "gemma-2-27b": 5,
  "gemma-3-27b": 2,
  "command-r": 4,
  "command-r-plus": 7,
  "command-r7b": 3,
  "command-a": 4,
  "qwen-2-72b": 5,
  "qwen-2-vl-7b": 3,
  "qwen-2.5-7b": 7,
  "qwen-2.5-72b": 7,
  "qwen-2.5-coder-32b": 8,
  "qwen-2.5-vl-72b": 5,
  "qwen-3-235b": 2,
  "qwen-3-32b": 6,
  "qwq-32b": 7,
  "deepseek-v3": 2,
  "deepseek-r1": 12,
  "deepseek-r1-distill-llama-70b": 5,
  "deepseek-r1-distill-qwen-1.5b": 3,
  "deepseek-r1-distill-qwen-14b": 3,
  "grok-2": 2,
  "grok-3": 3,
  "kimi-k2": 7,
  "sonar": 3,
  "sonar-pro": 3,
  "sonar-reasoning": 2,
  "sonar-reasoning-pro": 3,
  "r1-1776": 3,
  "nemotron-70b": 5,
  "sdxl-turbo": 5,
  "sd-3.5-large": 5,
  "flux": 8,
  "flux-pro": 3,
  "flux-dev": 8,
  "flux-schnell": 6,
  "flux-kontext": 2,
  "auto": 2,
  "gpt-5.2": 6,
  "gpt-5.1": 5,
  "gpt-5": 9,
  "gpt-5-thinking": 2,
  "gpt-5.4": 4,
  "gpt-5.4-mini": 4,
  "mercury": 2,
  "qwen-coder": 2,
  "mistral": 2,
  "gemini-3-flash": 8,
  "deepseek": 9,
  "gemma": 3,
  "grok": 3,
  "grok-4-20-reasoning": 2,
  "claude-opus-4.6": 3,
  "claude-opus-4.7": 3,
  "perplexity": 2,
  "kimi": 3,
  "nova": 2,
  "minimax-m2.7": 10,
  "minimax": 3,
  "mistral-large": 2,
  "step-3.5-flash": 8,
  "grok-4": 4,
  "grok-3-mini": 3,
  "qwen-3.7-plus": 4,
  "qwen-3.7-max": 4,
  "qwen-3.6-plus": 4,
  "qwen-3.6-max": 3,
  "qwen-3.6-27b": 6,
  "qwen-3.5-plus": 3,
  "qwen-3.6-35b-a3b": 7,
  "qwen-3.5-flash": 4,
  "qwen-3.5-397b-a17b": 8,
  "qwen-3.5-122b-a10b": 6,
  "qwen-3.5-27b": 6,
  "qwen-3.5-35b-a3b": 6,
  "qwen-3-max": 6,
  "qwen-plus": 4,
  "qwen-3-coder-plus": 3,
  "qwen-3-vl-plus": 2,
  "qwen-3-omni-flash": 3,
  "grok-4.1-fast": 2,
  "glm-5.2": 10,
  "kimi-k2.7-code": 10,
  "nvidia-nemotron-3-ultra-550b-a55b": 4,
  "nemotron-3-nano-omni-30b-a3b-reasoning": 3,
  "deepseek-v4-flash": 11,
  "deepseek-v4-pro": 11,
  "kimi-k2.6": 10,
  "mimo-v2.5": 5,
  "mimo-v2.5-pro": 7,
  "glm-5.1": 10,
  "gemma-4-26b-a4b-it": 6,
  "gemma-4-31b-it": 6,
  "nvidia-nemotron-3-super-120b-a12b": 2,
  "glm-5": 9,
  "minimax-m2.5": 9,
  "qwen-3-max-thinking": 4,
  "kimi-k2.5": 9,
  "glm-4.7-flash": 7,
  "deepseek-v3.2": 8,
  "qwythos-9b-claude-mythos-5-1m": 2,
  "vibethinker-3b": 2,
  "ornith-1.0-9b": 2,
  "fastcontext-1.0-4b-sft": 2,
  "qwable-9b-claude-fable-5": 2,
  "qwable-5-27b-coder": 2,
  "gemma-4-26b-a4b-styletune": 2,
  "nvidia-nemotron-3-ultra-550b-a55b-nvfp4": 3,
  "fastcontext-1.0-4b-rl": 2,
  "qwen-3-8b": 6,
  "qwen-3.6-27b-aeon-ultimate-uncensored": 2,
  "qwen-3-coder-30b-a3b": 6,
  "qwen-3-coder-next": 7,
  "llama-3.2-1b": 4,
  "z-image-engineer": 2,
  "llama-3.2-11b-vision": 4,
  "command-r-plus24": 5,
  "deepseek-r1-distill-qwen-32b": 3,
  "llama-3.1-nemotron-70b": 2,
  "krea-2-turbo": 2,
  "ideogram-4": 2,
  "z-image-turbo": 2,
  "ideogram.4.turbotime.lora": 2,
  "krea-2-lora-retroanime": 2,
  "llama-3": 4,
  "moonshotai/Kimi-K2-Instruct": 3,
  "qvq-72b": 3,
  "stable-diffusion-3.5-large": 3,
  "sdxl-1.0": 3,
  "claude-opus-4-6": 2,
  "claude-opus-4-7": 2,
  "gemini-3-pro": 2,
  "gpt-5.2-chat": 4,
  "claude-sonnet-4-6": 2,
  "claude-opus-4-5": 2,
  "claude-sonnet-4-5": 5,
  "gpt-5.3-chat": 4,
  "minimax-m3": 10,
  "claude-opus-4-1": 4,
  "glm-4.7": 7,
  "o3": 4,
  "gpt-5-chat": 4,
  "qwen-3-235b-a22b-2507": 5,
  "kimi-k2-0905": 5,
  "kimi-k2-0711": 2,
  "mistral-large-3": 4,
  "qwen-3-vl-235b-a22b": 6,
  "claude-opus-4": 5,
  "claude-haiku-4-5": 4,
  "mistral-medium-2508": 2,
  "qwen-3-next-80b-a3b": 6,
  "qwen-3-235b-a22b-thinking-2507": 4,
  "qwen-3-vl-235b-a22b-thinking": 4,
  "claude-sonnet-4": 5,
  "qwen-3-coder-480b-a35b": 5,
  "minimax-m2.1": 7,
  "qwen-3-30b-a3b-2507": 4,
  "qwen-3-235b-a22b": 6,
  "qwen-3-next-80b-a3b-thinking": 4,
  "trinity-large-thinking": 4,
  "gemma-3-27b-it": 5,
  "minimax-m1": 3,
  "mercury-2": 4,
  "minimax-m2": 6,
  "nova-2-lite": 4,
  "qwen-3-30b-a3b": 4,
  "gemma-3n-e4b-it": 4,
  "gpt-oss-20b": 7,
  "nvidia-nemotron-3-nano-30b-a3b": 2,
  "granite-4.1-8b": 3,
  "mistral-small-3.1-24b-2503": 2,
  "step-3.7-flash": 6,
  "ring-1t": 2,
  "mistral-medium-3.5": 2,
  "mistral-small-2603": 4,
  "qwen-3-vl-8b-thinking": 4,
  "fusion": 3,
  "claude-fable-5": 3,
  "qwen-vl-max": 2,
  "qwen-3-vl-8b": 6,
  "grok-4.3": 5,
  "gpt-5.5": 5,
  "claude-opus-4-8": 2,
  "glm-5v-turbo": 3,
  "qwen-image-2512": 2,
  "z-image": 2,
  "grok-imagine-image": 2,
  "qwen-max": 2,
  "qwen-turbo": 2,
  "qwen-3-14b": 5,
  "qwen-3-coder-flash": 2,
  "qwen-3-vl-30b-a3b": 5,
  "claude-3-5-sonnet": 2,
  "claude-3-7-sonnet": 2,
  "claude-3-haiku": 3,
  "gpt-5-codex": 3,
  "gpt-5-mini": 3,
  "gpt-5-nano": 4,
  "gpt-5.1-codex": 2,
  "gpt-5.1-codex-mini": 3,
  "gpt-5.2-codex": 3,
  "gpt-5.3-codex": 3,
  "gpt-5.4-nano": 4,
  "grok-4-1-fast-non-reasoning": 2,
  "grok-4-1-fast-reasoning": 2,
  "grok-4-20-non-reasoning": 2,
  "gemini-2.5-flash-lite": 4,
  "minimax-m2.7-highspeed": 2,
  "codestral-2508": 2,
  "devstral-2512": 2,
  "ministral-14b-2512": 2,
  "ministral-3b-2512": 2,
  "ministral-8b-2512": 2,
  "mistral-large-2512": 2,
  "mistral-medium-3-5": 2,
  "moonshot-v1-128k": 2,
  "moonshot-v1-128k-vision": 2,
  "moonshot-v1-32k": 2,
  "moonshot-v1-32k-vision": 2,
  "moonshot-v1-8k": 2,
  "moonshot-v1-8k-vision": 2,
  "gpt-5.1-chat": 3,
  "gpt-5.2-pro": 2,
  "gpt-5.4-pro": 3,
  "gpt-5.5-pro": 2,
  "o1-pro": 2,
  "o3-pro": 2,
  "jamba-large-1.7": 2,
  "aion-1.0": 2,
  "aion-1.0-mini": 2,
  "aion-2.0": 2,
  "aion-rp-llama-3.1-8b": 2,
  "olmo-3-32b-think": 2,
  "nova-lite": 2,
  "nova-micro": 3,
  "nova-premier": 2,
  "nova-pro": 2,
  "magnum-v4-72b": 2,
  "claude-opus-4.1": 2,
  "claude-opus-4.6-fast": 2,
  "claude-opus-4.7-fast": 2,
  "claude-opus-4.8-fast": 2,
  "coder-large": 2,
  "trinity-mini": 2,
  "virtuoso-large": 2,
  "ernie-4.5-vl-424b-a47b": 2,
  "seed-1.6": 2,
  "seed-1.6-flash": 2,
  "seed-2.0-lite": 2,
  "seed-2.0-mini": 2,
  "ui-tars-1.5-7b": 3,
  "dolphin-mistral-24b-venice-edition": 3,
  "command-r24": 5,
  "command-r7b24": 6,
  "north-mini-code": 2,
  "cogito-v2.1-671b": 2,
  "deepseek-chat-v3-0324": 2,
  "deepseek-chat-v3.1": 2,
  "deepseek-r1-0528": 3,
  "deepseek-v3.1-terminus": 4,
  "deepseek-v3.2-exp": 4,
  "gemini-2.5-flash-image": 2,
  "gemini-2.5-flash-lite-preview25": 2,
  "gemini-3-pro-image": 2,
  "gemini-3.1-flash-image": 2,
  "gemini-3.1-pro-preview-customtools": 3,
  "gemma-2-27b-it": 3,
  "gemma-3-12b-it": 4,
  "gemma-3-4b-it": 4,
  "lyria-3-clip": 2,
  "lyria-3-pro": 2,
  "mythomax-l2-13b": 2,
  "granite-4.0-h-micro": 2,
  "ling-2.6-1t": 3,
  "ling-2.6-flash": 2,
  "ring-2.6-1t": 2,
  "inflection-3-pi": 2,
  "inflection-3-productivity": 2,
  "kat-coder-pro": 2,
  "lfm-2-24b-a2b": 2,
  "lfm-2.5-1.2b": 2,
  "lfm-2.5-1.2b-thinking": 2,
  "weaver": 2,
  "phi-4": 4,
  "phi-4-mini": 3,
  "wizardlm-2-8x22b": 4,
  "minimax-01": 2,
  "minimax-m2-her": 2,
  "mistral-large-2407": 2,
  "mistral-medium-3.1": 2,
  "mistral-saba": 2,
  "mistral-small-24b-2501": 2,
  "mistral-small-3.2-24b": 2,
  "mixtral-8x22b": 2,
  "voxtral-small-24b-2507": 2,
  "kimi-k2-thinking": 5,
  "morph-v3-fast": 2,
  "morph-v3-large": 2,
  "nex-n2-pro": 2,
  "hermes-3-llama-3.1-405b": 2,
  "hermes-3-llama-3.1-70b": 2,
  "hermes-4-405b": 2,
  "hermes-4-70b": 3,
  "llama-3.3-nemotron-super-49b-v1.5": 2,
  "nemotron-3-nano-30b-a3b": 2,
  "nemotron-3-super-120b-a12b": 2,
  "nemotron-3-ultra-550b-a55b": 2,
  "nemotron-3.5-content-safety": 2,
  "nemotron-nano-12b-v2-vl": 2,
  "nemotron-nano-9b": 2,
  "gpt-3.5-turbo": 2,
  "gpt-3.5-turbo-0613": 2,
  "gpt-3.5-turbo-16k": 3,
  "gpt-4-turbo": 3,
  "gpt-4o-mini-search": 3,
  "gpt-4o-search": 3,
  "gpt-5-image": 2,
  "gpt-5-image-mini": 2,
  "gpt-5-pro": 2,
  "gpt-5.1-codex-max": 2,
  "gpt-5.4-image-2": 2,
  "gpt-audio": 4,
  "gpt-audio-mini": 3,
  "gpt-chat": 2,
  "gpt-oss-safeguard-20b": 4,
  "o3-deep-research": 3,
  "o4-mini-deep-research": 3,
  "bodybuilder": 2,
  "free": 2,
  "owl-alpha": 2,
  "pareto-code": 2,
  "perceptron-mk1": 2,
  "sonar-deep-research": 2,
  "sonar-pro-search": 2,
  "laguna-m.1": 2,
  "laguna-xs.2": 2,
  "qwen-3-30b-a3b-thinking-2507": 2,
  "qwen-3-coder": 5,
  "qwen-3-vl-30b-a3b-thinking": 3,
  "qwen-3-vl-32b": 3,
  "qwen-3.6-flash": 2,
  "reka-edge": 2,
  "reka-flash-3": 2,
  "relace-apply-3": 2,
  "relace-search": 2,
  "fugu-ultra": 2,
  "l3-lunaris-8b": 2,
  "l3.1-70b-hanami-x1": 2,
  "l3.1-euryale-70b": 2,
  "l3.3-euryale-70b": 2,
  "router": 2,
  "hunyuan-a13b": 2,
  "hy3": 2,
  "cydonia-24b-v4.1": 2,
  "rocinante-12b": 2,
  "skyfall-36b": 2,
  "unslopnemo-12b": 2,
  "remm-slerp-l2-13b": 2,
  "solar-pro-3": 2,
  "palmyra-x5": 2,
  "grok-4.20": 2,
  "grok-4.20-multi-agent": 3,
  "grok-build-0.1": 2,
  "gemini-flash": 2,
  "gemini-pro": 2,
  "gpt": 2,
  "gpt-mini": 2,
  "llama-guard-4-12b": 3,
  "qwen-3.5-9b": 4,
  "grok-2-vision": 2,
  "grok-3-fast": 2,
  "grok-3-mini-fast": 2,
  "grok-4-1-fast": 2,
  "grok-4-fast": 3,
  "grok-code-fast-1": 2,
  "glm-4.5": 5,
  "glm-4.5-air": 4,
  "glm-4.5v": 4,
  "glm-4.6": 6,
  "glm-4.6v": 3,
  "glm-4.6v-flash": 2,
  "glm-5-turbo": 2,
  "claude-3-opus": 2,
  "claude-3-sonnet": 2,
  "qwen-3-0.6b": 3,
  "qwen-3-1.7b": 3,
  "qwen-3-4b": 3,
  "qwen-2.5-coder-7b": 3,
  "deepseek-v3-0324": 4,
  "deepseek-r1-distill-llama-8b": 2,
  "deepseek-chat": 3,
  "deepseek-coder": 2,
  "glm-4": 2,
  "command-a25": 3,
  "command-r7b-arabic25": 3,
  "flux-kontext-dev": 2
}
parents = {
  "HuggingSpace": [
    "BlackForestLabs_Flux1Dev",
    "BlackForestLabs_Flux1KontextDev",
    "CohereForAI_C4AI_Command",
    "StabilityAI_SD35Large"
  ],
  "Copilot": [
    "CopilotAccount",
    "CopilotSession"
  ],
  "HuggingFace": [
    "HuggingFaceAPI",
    "HuggingFaceMedia"
  ],
  "MetaAI": [
    "MetaAIAccount"
  ],
  "OpenaiChat": [
    "OpenaiAccount"
  ],
  "PollinationsAI": [
    "PollinationsAudio",
    "PollinationsImage"
  ]
}
model_aliases = {
  "": "default",
  "Copilot": "dall-e-3",
  "openrouter:openai/gpt-4": "gpt-4",
  "openai/gpt-4": "gpt-4",
  "openrouter:openai/gpt-4o-2024-11-20": "gpt-4o",
  "openai/gpt-4o": "gpt-4o",
  "openrouter:openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini",
  "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini",
  "coral": "gpt-4o-mini-tts",
  "Think Deeper": "o1",
  "openai:openai/o1": "o1",
  "openai/o1": "o1",
  "openai:openai/o1-mini": "o1-mini",
  "openai:openai/o3-mini": "o3-mini",
  "o3-mini-2025-01-31": "o3-mini",
  "openai/o3-mini": "o3-mini",
  "openrouter:openai/o3-mini-high": "o3-mini-high",
  "openai/o3-mini-high": "o3-mini-high",
  "o4-mini-2025-04-16": "o4-mini",
  "openai:openai/o4-mini": "o4-mini",
  "openai/o4-mini": "o4-mini",
  "openrouter:openai/o4-mini-high": "o4-mini-high",
  "openai/o4-mini-high": "o4-mini-high",
  "gpt-4-1": "gpt-4.1",
  "gpt-4.1-2025-04-14": "gpt-4.1",
  "openai:openai/gpt-4.1": "gpt-4.1",
  "openai/gpt-4.1": "gpt-4.1",
  "gpt-4-1-mini": "gpt-4.1-mini",
  "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
  "openai:openai/gpt-4.1-mini": "gpt-4.1-mini",
  "openai/gpt-4.1-mini": "gpt-4.1-mini",
  "openai:openai/gpt-4.1-nano": "gpt-4.1-nano",
  "openai/gpt-4.1-nano": "gpt-4.1-nano",
  "gpt-4-5": "gpt-4.5",
  "openai:openai/gpt-4.5-preview": "gpt-4.5",
  "openai/gpt-oss-120b": "gpt-oss-120b",
  "togetherai:openai/gpt-oss-120b": "gpt-oss-120b",
  "gpt-oss:120b": "gpt-oss-120b",
  "Study": "study",
  "gptimage": "gpt-image",
  "openrouter:meta-llama/llama-2-70b-chat": "llama-2-70b",
  "openrouter:meta-llama/llama-3-8b-instruct": "llama-3-8b",
  "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b",
  "meta-llama/llama-3-8b-instruct": "llama-3-8b",
  "llama-3-8b-instruct": "llama-3-8b",
  "openrouter:meta-llama/llama-3-70b-instruct": "llama-3-70b",
  "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b",
  "llama-3-70b-instruct": "llama-3-70b",
  "meta/meta-llama-3-70b-instruct": "llama-3-70b",
  "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
  "llama3.1-8b": "llama-3.1-8b",
  "hf:meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
  "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
  "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
  "openrouter:meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
  "llama3.1-70b": "llama-3.1-70b",
  "hf:meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
  "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
  "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
  "hf:meta-llama/Llama-3.1-405B-Instruct": "llama-3.1-405b",
  "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
  "hf:meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
  "meta-llama/llama-3.2-3b-instruct": "llama-3.2-3b",
  "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b-vision",
  "openrouter:meta-llama/llama-3.2-90b-vision-instruct": "llama-3.2-90b",
  "meta-llama/Llama-3.3-70B-Instruct": "llama-3",
  "hf:meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
  "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
  "llama": "llama-",
  "meta-llama/llama-4-scout": "llama-4-scout",
  "meta-llama/llama-4-maverick": "llama-4-maverick",
  "mistralai/Mistral-Nemo-Instruct-2407": "mistral-nemo-2407",
  "mistral-nemo:latest": "mistral-nemo",
  "mistralai/mistral-nemo": "mistral-nemo",
  "mistral-small": "mistral-small-3.1-24b",
  "openrouter:mistralai/mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "mistralai/mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "openrouter:nousresearch/nous-hermes-2-mixtral-8x7b-dpo": "hermes-2-dpo",
  "microsoft/Phi-3.5-mini-instruct": "phi-3.5-mini",
  "openrouter:microsoft/phi-3.5-mini-128k-instruct": "phi-3.5-mini",
  "gemini-2.0-flash-001": "gemini-2.0-flash",
  "openrouter:google/gemini-2.5-flash-preview": "gemini-2.5-flash",
  "google/gemini-2.5-flash": "gemini-2.5-flash",
  "google/gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
  "openrouter:google/gemini-3.1-pro-preview": "gemini-3.1-pro",
  "google/gemini-3.1-pro-preview": "gemini-3.1-pro",
  "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "openrouter:google/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "google/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "openrouter:google/gemini-3.5-flash": "gemini-3.5-flash",
  "google/gemini-3.5-flash": "gemini-3.5-flash",
  "google/gemma-2-27b-it": "gemma-2-27b-it",
  "openrouter:google/gemma-2-27b-it": "gemma-2-27b-it",
  "command-r-08-2024": "command-r24",
  "command-r:35b": "command-r",
  "CohereForAI/c4ai-command-r-plus-08-2024": "command-r-plus24",
  "command-r-plus:104b": "command-r-plus",
  "openrouter:cohere/command-r7b-12-2024": "command-r7b24",
  "command-a-03-2025": "command-a25",
  "openrouter:cohere/command-a": "command-a",
  "cohere/command-a": "command-a",
  "Qwen/Qwen2-72B-Instruct": "qwen-2-72b",
  "openrouter:qwen/qwen-2-72b-instruct": "qwen-2-72b",
  "Qwen/Qwen2-VL-7B-Instruct": "qwen-2-vl-7b",
  "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b",
  "hf:Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b",
  "Qwen/Qwen2.5-7B": "qwen-2.5-7b",
  "qwen/qwen-2.5-7b-instruct": "qwen-2.5-7b",
  "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen-2.5-coder-32b",
  "hf:Qwen/Qwen2.5-72B-Instruct": "qwen-2.5-72b",
  "qwen/qwen-2.5-72b-instruct": "qwen-2.5-72b",
  "hf:Qwen/Qwen2.5-Coder-32B-Instruct": "qwen-2.5-coder-32b",
  "qwen/qwen-2.5-coder-32b-instruct": "qwen-2.5-coder-32b",
  "Qwen/Qwen2.5-VL-72B-Instruct": "qwen-2.5-vl-72b",
  "qwen/qwen2.5-vl-72b-instruct": "qwen-2.5-vl-72b",
  "qwen3-32b": "qwen-3-32b",
  "Qwen/Qwen3-32B": "qwen-3-32b",
  "qwen/qwen3-32b": "qwen-3-32b",
  "Qwen/QwQ-32B": "qwq-32b",
  "hf:Qwen/QwQ-32B-Preview": "qwq-32b",
  "openrouter:deepseek/deepseek-v3-base:free": "deepseek-v3",
  "deepseek-reasoning": "deepseek-r1",
  "deepseek-ai/DeepSeek-R1": "deepseek-r1",
  "deepseek-r1:7b": "deepseek-r1",
  "deepseek/deepseek-r1": "deepseek-r1",
  "v3": "deepseek",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
  "deepseek/deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
  "openrouter:deepseek/deepseek-r1-distill-qwen-1.5b": "deepseek-r1-distill-qwen-1.5b",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-r1-distill-qwen-1.5b",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-r1-distill-qwen-14b",
  "x-ai:x-ai/grok-3": "grok-3",
  "openrouter:moonshotai/kimi-k2": "kimi-k2",
  "kimi-k2:1t-cloud": "kimi-k2",
  "moonshotai/kimi-k2": "kimi-k2",
  "openrouter:perplexity/sonar": "sonar",
  "perplexity/sonar": "sonar",
  "openrouter:perplexity/sonar-pro": "sonar-pro",
  "perplexity/sonar-pro": "sonar-pro",
  "openrouter:perplexity/sonar-reasoning": "sonar-reasoning",
  "openrouter:perplexity/sonar-reasoning-pro": "sonar-reasoning-pro",
  "perplexity/sonar-reasoning-pro": "sonar-reasoning-pro",
  "openrouter:perplexity/r1-1776": "r1-1776",
  "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "llama-3.1-nemotron-70b",
  "openrouter:nvidia/llama-3.1-nemotron-70b-instruct": "nemotron-70b",
  "stabilityai/sdxl-turbo": "sdxl-turbo",
  "turbo": "sdxl-turbo",
  "stabilityai/stable-diffusion-3.5-large": "stable-diffusion-3.5-large",
  "stabilityai-stable-diffusion-3-5-large": "sd-3.5-large",
  "black-forest-labs/FLUX.1-dev": "flux-dev",
  "black-forest-labs-flux-1-dev": "flux-dev",
  "black-forest-labs/FLUX.1-schnell": "flux-schnell",
  "kontext": "flux-kontext",
  "openrouter/auto": "auto",
  "gpt-5-2": "gpt-5.2",
  "openai:openai/gpt-5.2": "gpt-5.2",
  "openai/gpt-5.2": "gpt-5.2",
  "gpt-5-2-instant": "gpt-5.2-instant",
  "gpt-5-2-thinking": "gpt-5.2-thinking",
  "gpt-5-1": "gpt-5.1",
  "azure:openai/gpt-5.1": "gpt-5.1",
  "openai/gpt-5.1": "gpt-5.1",
  "gpt-5-1-instant": "gpt-5.1-instant",
  "gpt-5-1-thinking": "gpt-5.1-thinking",
  "GPT-5": "gpt-5",
  "gpt-5-free": "gpt-5",
  "openai:openai/gpt-5": "gpt-5",
  "openai/gpt-5": "gpt-5",
  "openai:openai/gpt-5.4": "gpt-5.4",
  "openai/gpt-5.4": "gpt-5.4",
  "azure:openai/gpt-5.4-mini": "gpt-5.4-mini",
  "openai/gpt-5.4-mini": "gpt-5.4-mini",
  "qwen-coder:14b": "qwen-coder",
  "mistral:7b-instruct-v0.3-q4_0": "mistral",
  "openrouter:google/gemini-3-flash-preview": "gemini-3-flash",
  "gemini-3-flash-preview": "gemini-3-flash",
  "gemini-3-flash-preview:cloud": "gemini-3-flash",
  "google/gemini-3-flash-preview": "gemini-3-flash",
  "hf:deepseek-ai/DeepSeek-V3": "deepseek",
  "deepseek-ai/DeepSeek-V3": "deepseek",
  "deepseek-v2:16b-lite-chat-q8_0": "deepseek",
  "gemma:2b-instruct": "gemma",
  "grok-latest": "grok",
  "x-ai:x-ai/grok-4-20-reasoning": "grok-4-20-reasoning",
  "anthropic/claude-opus-4.6": "claude-opus-4.6",
  "anthropic/claude-opus-4.7": "claude-opus-4.7",
  "openrouter:~moonshotai/kimi-latest": "kimi",
  "~moonshotai/kimi-latest": "kimi",
  "minimax:minimax/minimax-m2.7": "minimax-m2.7",
  "MiniMaxAI/MiniMax-M2.7": "minimax-m2.7",
  "MiniMax-M2.7": "minimax-m2.7",
  "minimax-m2.7:cloud": "minimax-m2.7",
  "minimax/minimax-m2.7": "minimax-m2.7",
  "openrouter:minimax/minimax-01": "minimax-01",
  "mistralai/mistral-large": "mistral-large",
  "stepfun-ai/Step-3.5-Flash": "step-3.5-flash",
  "openrouter:stepfun/step-3.5-flash": "step-3.5-flash",
  "step-3.5-flash:free": "step-3.5-flash",
  "stepfun/step-3.5-flash": "step-3.5-flash",
  "x-ai:x-ai/grok-4": "grok-4",
  "openrouter:x-ai/grok-3-mini-beta": "grok-3-mini",
  "qwen3.7-plus": "qwen-3.7-plus",
  "togetherai:qwen/qwen3.7-plus": "qwen-3.7-plus",
  "qwen/qwen3.7-plus": "qwen-3.7-plus",
  "qwen3.7-max": "qwen-3.7-max",
  "togetherai:qwen/qwen3.7-max": "qwen-3.7-max",
  "qwen/qwen3.7-max": "qwen-3.7-max",
  "qwen3.6-plus-preview": "qwen-3.6-plus",
  "qwen3.6-plus": "qwen-3.6-plus",
  "alibaba:qwen/qwen3.6-plus": "qwen-3.6-plus",
  "qwen/qwen3.6-plus": "qwen-3.6-plus",
  "qwen3.6-max-preview": "qwen-3.6-max",
  "alibaba:qwen/qwen3.6-max-preview": "qwen-3.6-max",
  "qwen/qwen3.6-max-preview": "qwen-3.6-max",
  "qwen3.6-27b": "qwen-3.6-27b",
  "alibaba:qwen/qwen3.6-27b": "qwen-3.6-27b",
  "Qwen/Qwen3.6-27B": "qwen-3.6-27b",
  "qwen/qwen3.6-27b": "qwen-3.6-27b",
  "qwen-latest-series-invite-beta-v16": "qwen-series-invite-beta",
  "qwen3.5-plus": "qwen-3.5-plus",
  "openrouter:qwen/qwen3.5-plus-20260420": "qwen-3.5-plus",
  "qwen/qwen3.5-plus-02-15": "qwen-3.5-plus",
  "qwen3.5-omni-plus": "qwen-3.5-omni-plus",
  "qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "Qwen/Qwen3.6-35B-A3B": "qwen-3.6-35b-a3b",
  "alibaba:qwen/qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "qwen/qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "qwen3.5-flash": "qwen-3.5-flash",
  "openrouter:qwen/qwen3.5-flash-02-23": "qwen-3.5-flash",
  "qwen/qwen3.5-flash-02-23": "qwen-3.5-flash",
  "qwen3.5-max-2026-03-08": "qwen-3.5-max",
  "qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "Qwen/Qwen3.5-397B-A17B": "qwen-3.5-397b-a17b",
  "alibaba:qwen/qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "qwen/qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "alibaba:qwen/qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "Qwen/Qwen3.5-122B-A10B": "qwen-3.5-122b-a10b",
  "qwen/qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "qwen3.5-omni-flash": "qwen-3.5-omni-flash",
  "qwen3.5-27b": "qwen-3.5-27b",
  "alibaba:qwen/qwen3.5-27b": "qwen-3.5-27b",
  "Qwen/Qwen3.5-27B": "qwen-3.5-27b",
  "qwen/qwen3.5-27b": "qwen-3.5-27b",
  "qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "alibaba:qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "Qwen/Qwen3.5-35B-A3B": "qwen-3.5-35b-a3b",
  "qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "qwen3-max-2026-01-23": "qwen-3-max",
  "Qwen/Qwen3-Max": "qwen-3-max",
  "qwen3-max-2025-09-26": "qwen-3-max",
  "alibaba:qwen/qwen3-max": "qwen-3-max",
  "qwen3-max": "qwen-3-max",
  "qwen/qwen3-max": "qwen-3-max",
  "qwen-plus-2025-07-28": "qwen-plus",
  "openrouter:qwen/qwen-plus": "qwen-plus",
  "qwen/qwen-plus": "qwen-plus",
  "qwen3-coder-plus": "qwen-3-coder-plus",
  "alibaba:qwen/qwen3-coder-plus": "qwen-3-coder-plus",
  "qwen/qwen3-coder-plus": "qwen-3-coder-plus",
  "qwen3-vl-plus": "qwen-3-vl-plus",
  "alibaba:qwen/qwen3-vl-plus": "qwen-3-vl-plus",
  "qwen3-omni-flash-2025-12-01": "qwen-3-omni-flash",
  "qwen3-omni-flash": "qwen-3-omni-flash",
  "alibaba:qwen/qwen3-omni-flash": "qwen-3-omni-flash",
  "gpt-4o-mini-image-free": "gpt-4o-mini-image",
  "grok-4.1-fast-free": "grok-4.1-fast",
  "openrouter-free": "openrouter",
  "zai-org/GLM-5.2": "glm-5.2",
  "zai-org/GLM-5.2-FP8": "glm-5.2",
  "z-ai:z-ai/glm-5.2": "glm-5.2",
  "glm-5.2:cloud": "glm-5.2",
  "z-ai/glm-5.2": "glm-5.2",
  "moonshotai/Kimi-K2.7-Code": "kimi-k2.7-code",
  "togetherai:moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "kimi-k2.7-code:cloud": "kimi-k2.7-code",
  "moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": "nvidia-nemotron-3-ultra-550b-a55b",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia-nemotron-3-ultra-550b-a55b",
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning": "nemotron-3-nano-omni-30b-a3b-reasoning",
  "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "nemotron-3-nano-omni-30b-a3b-reasoning",
  "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "nemotron-3-nano-omni-30b-a3b-reasoning",
  "deepseek-ai/DeepSeek-V4-Flash": "deepseek-v4-flash",
  "deepseek:deepseek/deepseek-v4-flash": "deepseek-v4-flash",
  "deepseek-v4-flash:cloud": "deepseek-v4-flash",
  "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
  "deepseek-ai/DeepSeek-V4-Pro": "deepseek-v4-pro",
  "deepseek:deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "deepseek-v4-pro:cloud": "deepseek-v4-pro",
  "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "moonshotai/Kimi-K2.6": "kimi-k2.6",
  "moonshotai:moonshotai/kimi-k2.6": "kimi-k2.6",
  "kimi-k2.6:cloud": "kimi-k2.6",
  "moonshotai/kimi-k2.6": "kimi-k2.6",
  "XiaomiMiMo/MiMo-V2.5": "mimo-v2.5",
  "openrouter:xiaomi/mimo-v2.5": "mimo-v2.5",
  "xiaomi/mimo-v2.5": "mimo-v2.5",
  "XiaomiMiMo/MiMo-V2.5-Pro": "mimo-v2.5-pro",
  "openrouter:xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
  "xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
  "zai-org/GLM-5.1": "glm-5.1",
  "z-ai:z-ai/glm-5.1": "glm-5.1",
  "zai-org/GLM-5.1-FP8": "glm-5.1",
  "glm-5.1:cloud": "glm-5.1",
  "z-ai/glm-5.1": "glm-5.1",
  "google/gemma-4-26B-A4B-it": "gemma-4-26b-a4b-it",
  "openrouter:google/gemma-4-26b-a4b-it:free": "gemma-4-26b-a4b-it",
  "google/gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
  "google/gemma-4-31B-it": "gemma-4-31b-it",
  "togetherai:google/gemma-4-31b-it": "gemma-4-31b-it",
  "google/gemma-4-31b-it": "gemma-4-31b-it",
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": "nvidia-nemotron-3-super-120b-a12b",
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia-nemotron-3-super-120b-a12b",
  "zai-org/GLM-5": "glm-5",
  "z-ai:z-ai/glm-5": "glm-5",
  "glm-5:cloud": "glm-5",
  "z-ai/glm-5": "glm-5",
  "MiniMaxAI/MiniMax-M2.5": "minimax-m2.5",
  "minimax:minimax/minimax-m2.5": "minimax-m2.5",
  "minimax-m2.5:cloud": "minimax-m2.5",
  "minimax/minimax-m2.5": "minimax-m2.5",
  "Qwen/Qwen3-Max-Thinking": "qwen-3-max-thinking",
  "qwen3-max-thinking": "qwen-3-max-thinking",
  "openrouter:qwen/qwen3-max-thinking": "qwen-3-max-thinking",
  "qwen/qwen3-max-thinking": "qwen-3-max-thinking",
  "moonshotai/Kimi-K2.5": "kimi-k2.5",
  "moonshotai:moonshotai/kimi-k2.5": "kimi-k2.5",
  "kimi-k2.5:cloud": "kimi-k2.5",
  "moonshotai/kimi-k2.5": "kimi-k2.5",
  "zai-org/GLM-4.7-Flash": "glm-4.7-flash",
  "z-ai:z-ai/glm-4.7-flash": "glm-4.7-flash",
  "glm-4.7-flash:latest": "glm-4.7-flash",
  "z-ai/glm-4.7-flash": "glm-4.7-flash",
  "deepseek-ai/DeepSeek-V3.2": "deepseek-v3.2",
  "openrouter:deepseek/deepseek-v3.2": "deepseek-v3.2",
  "deepseek-v3.2:cloud": "deepseek-v3.2",
  "deepseek/deepseek-v3.2": "deepseek-v3.2",
  "black-forest-labs/FLUX-2-klein-4b": "flux-2-klein-4b",
  "black-forest-labs/FLUX-2-klein-9b": "flux-2-klein-9b",
  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M": "qwythos-9b-claude-mythos-5-1m",
  "WeiboAI/VibeThinker-3B": "vibethinker-3b",
  "deepreinforce-ai/Ornith-1.0-9B": "ornith-1.0-9b",
  "microsoft/FastContext-1.0-4B-SFT": "fastcontext-1.0-4b-sft",
  "empero-ai/Qwable-9B-Claude-Fable-5": "qwable-9b-claude-fable-5",
  "DJLougen/Qwable-5-27B-Coder": "qwable-5-27b-coder",
  "Gryphe/Gemma-4-26B-A4B-StyleTune-V2": "gemma-4-26b-a4b-styletune",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "nvidia-nemotron-3-ultra-550b-a55b-nvfp4",
  "microsoft/FastContext-1.0-4B-RL": "fastcontext-1.0-4b-rl",
  "Qwen/Qwen3-8B": "qwen-3-8b",
  "qwen3-8b": "qwen-3-8b",
  "qwen/qwen3-8b": "qwen-3-8b",
  "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16": "qwen-3.6-27b-aeon-ultimate-uncensored",
  "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen-3-coder-30b-a3b",
  "alibaba:qwen/qwen3-coder-30b-a3b-instruct": "qwen-3-coder-30b-a3b",
  "qwen3-coder-30b-a3b": "qwen-3-coder-30b-a3b",
  "qwen/qwen3-coder-30b-a3b-instruct": "qwen-3-coder-30b-a3b",
  "Qwen/Qwen3-Coder-Next": "qwen-3-coder-next",
  "openrouter:qwen/qwen3-coder-next": "qwen-3-coder-next",
  "qwen3-coder-next": "qwen-3-coder-next",
  "qwen3-coder-next:q8_0": "qwen-3-coder-next",
  "qwen/qwen3-coder-next": "qwen-3-coder-next",
  "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
  "meta-llama/llama-3.2-1b-instruct": "llama-3.2-1b",
  "BennyDaBall/Z-Image-Engineer-V6": "z-image-engineer",
  "openrouter:meta-llama/llama-3.2-11b-vision-instruct": "llama-3.2-11b-vision",
  "hf:meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b-vision",
  "meta-llama/llama-3.2-11b-vision-instruct": "llama-3.2-11b-vision",
  "openrouter:cohere/command-r-plus-08-2024": "command-r-plus24",
  "command-r-plus-08-2024": "command-r-plus24",
  "cohere/command-r-plus-08-2024": "command-r-plus24",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-distill-qwen-32b",
  "hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "llama-3.1-nemotron-70b",
  "krea/Krea-2-Turbo": "krea-2-turbo",
  "krea/Krea-2-Turbo:fal-ai": "krea-2-turbo",
  "krea/Krea-2-Raw": "krea-2-raw",
  "AlperKTS/Krea2_FP8": "krea2.fp8",
  "ideogram-ai/ideogram-4-fp8": "ideogram-4",
  "ideogram-ai/ideogram-4-fp8:fal-ai": "ideogram-4",
  "Tongyi-MAI/Z-Image-Turbo": "z-image-turbo",
  "Tongyi-MAI/Z-Image-Turbo:wavespeed": "z-image-turbo",
  "vantagewithai/Krea-2-Turbo-GGUF": "krea-2-turbo-gguf",
  "ostris/ideogram_4_turbotime_lora": "ideogram.4.turbotime.lora",
  "ostris/ideogram_4_turbotime_lora:fal-ai": "ideogram.4.turbotime.lora",
  "krea/Krea-2-LoRA-retroanime": "krea-2-lora-retroanime",
  "krea/Krea-2-LoRA-retroanime:fal-ai": "krea-2-lora-retroanime",
  "ideogram-ai/ideogram-4-nf4": "ideogram-4-nf4",
  "Phr00t/Qwen-Image-Edit-Rapid-AIO": "qwen-image-edit-rapid-aio",
  "gokaygokay/Krea-2-Realism-LoRA": "krea-2-realism-lora",
  "ponpoke/flux2-klein-9b-uncensored-text-encoder": "flux2-klein-9b-uncensored-text-encoder",
  "llama3:8b-instruct-q4_K_M": "llama-3",
  "moonshotai/Kimi-K2-Instruct-0905": "kimi-k2-0905",
  "Qwen/QVQ-72B-Preview": "qvq-72b",
  "stabilityai/stable-diffusion-xl-base-1.0": "sdxl-1.0",
  "Wan-AI/Wan2.2-TI2V-5B:wavespeed": "wan2.2-ti2v-5b",
  "meituan-longcat/LongCat-Video:fal-ai": "longcat-video",
  "Wan-AI/Wan2.1-T2V-1.3B:wavespeed": "wan2.1-t2v-1.3b",
  "Wan-AI/Wan2.1-T2V-14B:wavespeed": "wan2.1-t2v-14b",
  "tencent/HunyuanVideo-1.5:wavespeed": "hunyuanvideo-1.5",
  "Wan-AI/Wan2.2-T2V-A14B:replicate": "wan2.2-t2v-a14b",
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers:replicate": "wan2.2-t2v-a14b-diffusers",
  "zai-org/CogVideoX-5b:fal-ai": "cogvideox-5b",
  "genmo/mochi-1-preview:fal-ai": "mochi-1",
  "Max": "max",
  "anthropic:anthropic/claude-opus-4-6": "claude-opus-4-6",
  "anthropic:anthropic/claude-opus-4-7": "claude-opus-4-7",
  "gpt-5.2-chat-latest": "gpt-5.2-chat",
  "openai:openai/gpt-5.2-chat": "gpt-5.2-chat",
  "openai/gpt-5.2-chat": "gpt-5.2-chat",
  "anthropic:anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
  "claude-opus-4-5-20251101": "claude-opus-4-5",
  "anthropic:anthropic/claude-opus-4-5": "claude-opus-4-5",
  "ernie-5.1-preview": "ernie-5.1",
  "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
  "anthropic:anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
  "claude-sonnet-4.5": "claude-sonnet-4-5",
  "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5",
  "gpt-5.3-chat-latest": "gpt-5.3-chat",
  "openrouter:openai/gpt-5.3-chat": "gpt-5.3-chat",
  "openai/gpt-5.3-chat": "gpt-5.3-chat",
  "minimax:minimax/minimax-m3": "minimax-m3",
  "MiniMaxAI/MiniMax-M3": "minimax-m3",
  "MiniMax-M3": "minimax-m3",
  "minimax-m3:cloud": "minimax-m3",
  "minimax/minimax-m3": "minimax-m3",
  "claude-opus-4-1-20250805": "claude-opus-4-1",
  "anthropic:anthropic/claude-opus-4-1": "claude-opus-4-1",
  "claude-opus-4-1-latest": "claude-opus-4-1",
  "z-ai:z-ai/glm-4.7": "glm-4.7",
  "zai-org/GLM-4.7": "glm-4.7",
  "glm-4.7:cloud": "glm-4.7",
  "z-ai/glm-4.7": "glm-4.7",
  "o3-2025-04-16": "o3",
  "openai:openai/o3": "o3",
  "openai/o3": "o3",
  "openai:openai/gpt-5-chat": "gpt-5-chat",
  "openai/gpt-5-chat": "gpt-5-chat",
  "qwen3-235b-a22b-instruct-2507": "qwen-3-235b-a22b-2507",
  "togetherai:qwen/qwen3-235b-a22b-instruct-2507-tput": "qwen-3-235b-a22b-2507",
  "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen-3-235b-a22b-2507",
  "qwen/qwen3-235b-a22b-2507": "qwen-3-235b-a22b-2507",
  "kimi-k2-0905-preview": "kimi-k2-0905",
  "openrouter:moonshotai/kimi-k2-0905": "kimi-k2-0905",
  "moonshotai/kimi-k2-0905": "kimi-k2-0905",
  "kimi-k2-0711-preview": "kimi-k2-0711",
  "mistral-large-3:675b": "mistral-large-3",
  "mistral-large-3:675b-cloud": "mistral-large-3",
  "qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "openrouter:qwen/qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "qwen3-vl-235b-a22b": "qwen-3-vl-235b-a22b",
  "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen-3-vl-235b-a22b",
  "qwen/qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "claude-opus-4-20250514": "claude-opus-4",
  "anthropic:anthropic/claude-opus-4": "claude-opus-4",
  "claude-opus-4-20250522": "claude-opus-4",
  "anthropic/claude-opus-4": "claude-opus-4",
  "claude-haiku-4-5-20251001": "claude-haiku-4-5",
  "anthropic:anthropic/claude-haiku-4-5": "claude-haiku-4-5",
  "claude-haiku-4.5": "claude-haiku-4-5",
  "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
  "mistralai:mistralai/mistral-medium-2508": "mistral-medium-2508",
  "qwen3-235b-a22b-no-thinking": "qwen-3-235b-a22b-no-thinking",
  "qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "openrouter:qwen/qwen3-next-80b-a3b-instruct:free": "qwen-3-next-80b-a3b",
  "qwen3-next-80b-a3b": "qwen-3-next-80b-a3b",
  "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen-3-next-80b-a3b",
  "qwen/qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "Qwen/Qwen3-235B-A22B-Thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "qwen/qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "openrouter:qwen/qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "Qwen/Qwen3-VL-235B-A22B-Thinking": "qwen-3-vl-235b-a22b-thinking",
  "qwen/qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "claude-sonnet-4-20250514": "claude-sonnet-4",
  "anthropic:anthropic/claude-sonnet-4": "claude-sonnet-4",
  "claude-sonnet-4-latest": "claude-sonnet-4",
  "anthropic/claude-sonnet-4": "claude-sonnet-4",
  "qwen3-coder-480b-a35b-instruct": "qwen-3-coder-480b-a35b",
  "alibaba:qwen/qwen3-coder-480b-a35b-instruct": "qwen-3-coder-480b-a35b",
  "qwen3-coder-480b-a35b": "qwen-3-coder-480b-a35b",
  "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen-3-coder-480b-a35b",
  "minimax-m2.1-preview": "minimax-m2.1",
  "minimax:minimax/minimax-m2.1": "minimax-m2.1",
  "MiniMaxAI/MiniMax-M2.1": "minimax-m2.1",
  "minimax-m2.1:cloud": "minimax-m2.1",
  "minimax/minimax-m2.1": "minimax-m2.1",
  "qwen3-30b-a3b-instruct-2507": "qwen-3-30b-a3b-2507",
  "openrouter:qwen/qwen3-30b-a3b-instruct-2507": "qwen-3-30b-a3b-2507",
  "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen-3-30b-a3b-2507",
  "qwen/qwen3-30b-a3b-instruct-2507": "qwen-3-30b-a3b-2507",
  "qwen3-235b-a22b": "qwen-3-235b-a22b",
  "alibaba:qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "Qwen/Qwen3-235B-A22B": "qwen-3-235b-a22b",
  "qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "qwen3-next-80b-a3b-thinking": "qwen-3-next-80b-a3b-thinking",
  "alibaba:qwen/qwen3-next-80b-a3b-thinking": "qwen-3-next-80b-a3b-thinking",
  "Qwen/Qwen3-Next-80B-A3B-Thinking": "qwen-3-next-80b-a3b-thinking",
  "qwen/qwen3-next-80b-a3b-thinking": "qwen-3-next-80b-a3b-thinking",
  "openrouter:arcee-ai/trinity-large-thinking": "trinity-large-thinking",
  "arcee-ai/Trinity-Large-Thinking": "trinity-large-thinking",
  "arcee-ai/trinity-large-thinking": "trinity-large-thinking",
  "openrouter:google/gemma-3-27b-it": "gemma-3-27b-it",
  "google/gemma-3-27b-it": "gemma-3-27b-it",
  "openrouter:minimax/minimax-m1": "minimax-m1",
  "minimax/minimax-m1": "minimax-m1",
  "openrouter:inception/mercury-2": "mercury-2",
  "inception/mercury-2": "mercury-2",
  "minimax-m2-preview": "minimax-m2",
  "minimax:minimax/minimax-m2": "minimax-m2",
  "MiniMaxAI/MiniMax-M2": "minimax-m2",
  "minimax-m2:cloud": "minimax-m2",
  "minimax/minimax-m2": "minimax-m2",
  "openrouter:amazon/nova-2-lite-v1": "nova-2-lite",
  "amazon/nova-2-lite-v1": "nova-2-lite",
  "olmo-3.1-32b-instruct": "olmo-3.1-32b",
  "qwen3-30b-a3b": "qwen-3-30b-a3b",
  "openrouter:qwen/qwen3-30b-a3b": "qwen-3-30b-a3b",
  "qwen/qwen3-30b-a3b": "qwen-3-30b-a3b",
  "togetherai:google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
  "google/gemma-3n-E4B-it": "gemma-3n-e4b-it",
  "google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
  "togetherai:openai/gpt-oss-20b": "gpt-oss-20b",
  "openai/gpt-oss-20b": "gpt-oss-20b",
  "gpt-oss:20b": "gpt-oss-20b",
  "nvidia-nemotron-3-nano-30b-a3b-bf16": "nvidia-nemotron-3-nano-30b-a3b",
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia-nemotron-3-nano-30b-a3b",
  "openrouter:ibm-granite/granite-4.1-8b": "granite-4.1-8b",
  "ibm-granite/granite-4.1-8b": "granite-4.1-8b",
  "mistral-small-3.1-24b-instruct-2503": "mistral-small-3.1-24b-2503",
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-3.1-24b-2503",
  "jebel_1": "jebel.1",
  "jebel_2": "jebel.2",
  "artemis_1": "artemis.1",
  "artemis_2": "artemis.2",
  "openrouter:stepfun/step-3.7-flash": "step-3.7-flash",
  "stepfun-ai/Step-3.7-Flash": "step-3.7-flash",
  "stepfun/step-3.7-flash": "step-3.7-flash",
  "globe_2": "globe.2",
  "hofburg_2": "hofburg.2",
  "hofburg_3": "hofburg.3",
  "nightride-on-v2": "nightride-on",
  "inclusionAI/Ring-1T": "ring-1t",
  "u2-preview": "u2",
  "stephen-v2": "stephen",
  "EB45-vision": "eb45-vision",
  "EB45-turbo": "eb45-turbo",
  "mistral-medium-3.5:latest": "mistral-medium-3.5",
  "hofburg_4": "hofburg.4",
  "hofburg_5": "hofburg.5",
  "mistralai:mistralai/mistral-small-2603": "mistral-small-2603",
  "mistralai/mistral-small-2603": "mistral-small-2603",
  "qwen3-vl-8b-thinking": "qwen-3-vl-8b-thinking",
  "openrouter:qwen/qwen3-vl-8b-thinking": "qwen-3-vl-8b-thinking",
  "Qwen/Qwen3-VL-8B-Thinking": "qwen-3-vl-8b-thinking",
  "qwen/qwen3-vl-8b-thinking": "qwen-3-vl-8b-thinking",
  "openrouter:openrouter/fusion": "fusion",
  "openrouter/fusion": "fusion",
  "anthropic:anthropic/claude-fable-5": "claude-fable-5",
  "anthropic/claude-fable-5": "claude-fable-5",
  "artemis_3": "artemis.3",
  "artemis_4": "artemis.4",
  "qwen-vl-max-2025-08-13": "qwen-vl-max",
  "openrouter:qwen/qwen-vl-max": "qwen-vl-max",
  "qwen3-vl-8b-instruct": "qwen-3-vl-8b",
  "openrouter:qwen/qwen3-vl-8b-instruct": "qwen-3-vl-8b",
  "qwen3-vl-8b": "qwen-3-vl-8b",
  "Qwen/Qwen3-VL-8B-Instruct": "qwen-3-vl-8b",
  "qwen/qwen3-vl-8b-instruct": "qwen-3-vl-8b",
  "azure:x-ai/grok-4.3": "grok-4.3",
  "x-ai/grok-4.3": "grok-4.3",
  "glassy_lagoon": "glassy.lagoon",
  "emerald_lagoon": "emerald.lagoon",
  "openai:openai/gpt-5.5": "gpt-5.5",
  "openai/gpt-5.5": "gpt-5.5",
  "anthropic:anthropic/claude-opus-4-8": "claude-opus-4-8",
  "amazon.nova-pro-v1:0": "amazon.nova-pro",
  "z-ai:z-ai/glm-5v-turbo": "glm-5v-turbo",
  "z-ai/glm-5v-turbo": "glm-5v-turbo",
  "imagen-4.0-generate-001": "imagen-4.0-generate",
  "Qwen/Qwen-Image-2512:fal-ai": "qwen-image-2512",
  "wan2.5-preview": "wan2.5",
  "wan2.5-t2i-preview": "wan2.5-t2i",
  "recraft-v3": "recraft",
  "Tongyi-MAI/Z-Image:fal-ai": "z-image",
  "imagen-3.0-generate-002": "imagen-3.0-generate",
  "phantom_brush": "phantom.brush",
  "zen-bear-v4": "zen-bear",
  "auto-bear-v2": "auto-bear",
  "spectral_ink": "spectral.ink",
  "phantom_quill": "phantom.quill",
  "wan2.5-i2i-preview": "wan2.5-i2i",
  "imagen-4.0-ultra-generate-001": "imagen-4.0-ultra-generate",
  "imagen-4.0-fast-generate-001": "imagen-4.0-fast-generate",
  "chatgpt-image-latest-high-fidelity (20251216)": "chatgpt-image-high-fidelity (20251216)",
  "alibaba:qwen/qvq-max": "qvq-max",
  "alibaba:qwen/qwen-flash": "qwen-flash",
  "openrouter:qwen/qwen-max": "qwen-max",
  "alibaba:qwen/qwen-mt-plus": "qwen-mt-plus",
  "alibaba:qwen/qwen-mt-turbo": "qwen-mt-turbo",
  "alibaba:qwen/qwen-omni-turbo": "qwen-omni-turbo",
  "openrouter:qwen/qwen-turbo": "qwen-turbo",
  "alibaba:qwen/qwen-vl-ocr": "qwen-vl-ocr",
  "openrouter:qwen/qwen-vl-plus": "qwen-vl-plus",
  "alibaba:qwen/qwen2-5-14b-instruct": "qwen-2-5-14b",
  "alibaba:qwen/qwen2-5-32b-instruct": "qwen-2-5-32b",
  "alibaba:qwen/qwen2-5-72b-instruct": "qwen-2-5-72b",
  "alibaba:qwen/qwen2-5-7b-instruct": "qwen-2-5-7b",
  "alibaba:qwen/qwen2-5-omni-7b": "qwen-2-5-omni-7b",
  "alibaba:qwen/qwen2-5-vl-72b-instruct": "qwen-2-5-vl-72b",
  "alibaba:qwen/qwen2-5-vl-7b-instruct": "qwen-2-5-vl-7b",
  "qwen3-14b": "qwen-3-14b",
  "Qwen/Qwen3-14B": "qwen-3-14b",
  "qwen/qwen3-14b": "qwen-3-14b",
  "alibaba:qwen/qwen3-coder-flash": "qwen-3-coder-flash",
  "qwen/qwen3-coder-flash": "qwen-3-coder-flash",
  "openrouter:qwen/qwen3-vl-30b-a3b-instruct": "qwen-3-vl-30b-a3b",
  "qwen3-vl-30b-a3b": "qwen-3-vl-30b-a3b",
  "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen-3-vl-30b-a3b",
  "qwen/qwen3-vl-30b-a3b-instruct": "qwen-3-vl-30b-a3b",
  "alibaba:qwen/qwq-plus": "qwq-plus",
  "anthropic:anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet",
  "claude-3-5-sonnet-latest": "claude-3-5-sonnet",
  "anthropic:anthropic/claude-3-7-sonnet": "claude-3-7-sonnet",
  "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
  "claude-3-haiku-20240307": "claude-3-haiku",
  "anthropic/claude-3-haiku": "claude-3-haiku",
  "azure:openai/gpt-5-codex": "gpt-5-codex",
  "openai/gpt-5-codex": "gpt-5-codex",
  "openai:openai/gpt-5-mini": "gpt-5-mini",
  "openai/gpt-5-mini": "gpt-5-mini",
  "openai:openai/gpt-5-nano": "gpt-5-nano",
  "openai/gpt-5-nano": "gpt-5-nano",
  "azure:openai/gpt-5.1-codex": "gpt-5.1-codex",
  "openai/gpt-5.1-codex": "gpt-5.1-codex",
  "azure:openai/gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
  "openai/gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
  "azure:openai/gpt-5.2-codex": "gpt-5.2-codex",
  "openai/gpt-5.2-codex": "gpt-5.2-codex",
  "azure:openai/gpt-5.3-codex": "gpt-5.3-codex",
  "openai/gpt-5.3-codex": "gpt-5.3-codex",
  "azure:openai/gpt-5.4-nano": "gpt-5.4-nano",
  "openai/gpt-5.4-nano": "gpt-5.4-nano",
  "azure:x-ai/grok-4-1-fast-non-reasoning": "grok-4-1-fast-non-reasoning",
  "azure:x-ai/grok-4-1-fast-reasoning": "grok-4-1-fast-reasoning",
  "x-ai:x-ai/grok-4-20-non-reasoning": "grok-4-20-non-reasoning",
  "google:google/gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
  "openrouter:google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
  "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
  "minimax:minimax/minimax-m2.1-highspeed": "minimax-m2.1-highspeed",
  "minimax:minimax/minimax-m2.5-highspeed": "minimax-m2.5-highspeed",
  "minimax:minimax/minimax-m2.7-highspeed": "minimax-m2.7-highspeed",
  "MiniMax-M2.7-highspeed": "minimax-m2.7-highspeed",
  "mistralai:mistralai/codestral-2508": "codestral-2508",
  "mistralai/codestral-2508": "codestral-2508",
  "mistralai:mistralai/devstral-2512": "devstral-2512",
  "mistralai/devstral-2512": "devstral-2512",
  "mistralai:mistralai/magistral-medium-2509": "magistral-medium-2509",
  "mistralai:mistralai/magistral-small-2509": "magistral-small-2509",
  "mistralai:mistralai/ministral-14b-2512": "ministral-14b-2512",
  "mistralai/ministral-14b-2512": "ministral-14b-2512",
  "mistralai:mistralai/ministral-3b-2512": "ministral-3b-2512",
  "mistralai/ministral-3b-2512": "ministral-3b-2512",
  "mistralai:mistralai/ministral-8b-2512": "ministral-8b-2512",
  "mistralai/ministral-8b-2512": "ministral-8b-2512",
  "mistralai:mistralai/mistral-large-2512": "mistral-large-2512",
  "mistralai/mistral-large-2512": "mistral-large-2512",
  "mistralai:mistralai/mistral-medium-3-5": "mistral-medium-3-5",
  "mistralai/mistral-medium-3-5": "mistral-medium-3-5",
  "mistralai:mistralai/open-mistral-nemo-2407": "open-mistral-nemo-2407",
  "mistralai:mistralai/voxtral-small-2507": "voxtral-small-2507",
  "moonshotai:moonshotai/moonshot-v1-128k": "moonshot-v1-128k",
  "moonshotai:moonshotai/moonshot-v1-128k-vision-preview": "moonshot-v1-128k-vision",
  "moonshotai:moonshotai/moonshot-v1-32k": "moonshot-v1-32k",
  "moonshotai:moonshotai/moonshot-v1-32k-vision-preview": "moonshot-v1-32k-vision",
  "moonshotai:moonshotai/moonshot-v1-8k": "moonshot-v1-8k",
  "moonshotai:moonshotai/moonshot-v1-8k-vision-preview": "moonshot-v1-8k-vision",
  "moonshotai:moonshotai/moonshot-v1-auto": "moonshot-v1-auto",
  "openai:openai/gpt-5.1-chat": "gpt-5.1-chat",
  "openai/gpt-5.1-chat": "gpt-5.1-chat",
  "openai:openai/gpt-5.2-pro": "gpt-5.2-pro",
  "openai/gpt-5.2-pro": "gpt-5.2-pro",
  "openai:openai/gpt-5.4-pro": "gpt-5.4-pro",
  "openai/gpt-5.4-pro": "gpt-5.4-pro",
  "openai:openai/gpt-5.5-pro": "gpt-5.5-pro",
  "openai/gpt-5.5-pro": "gpt-5.5-pro",
  "openai:openai/o1-pro": "o1-pro",
  "openai/o1-pro": "o1-pro",
  "openai:openai/o3-pro": "o3-pro",
  "openai/o3-pro": "o3-pro",
  "openrouter:ai21/jamba-large-1.7": "jamba-large-1.7",
  "ai21/jamba-large-1.7": "jamba-large-1.7",
  "openrouter:aion-labs/aion-1.0": "aion-1.0",
  "aion-labs/aion-1.0": "aion-1.0",
  "openrouter:aion-labs/aion-1.0-mini": "aion-1.0-mini",
  "aion-labs/aion-1.0-mini": "aion-1.0-mini",
  "openrouter:aion-labs/aion-2.0": "aion-2.0",
  "aion-labs/aion-2.0": "aion-2.0",
  "openrouter:aion-labs/aion-rp-llama-3.1-8b": "aion-rp-llama-3.1-8b",
  "aion-labs/aion-rp-llama-3.1-8b": "aion-rp-llama-3.1-8b",
  "openrouter:allenai/olmo-3-32b-think": "olmo-3-32b-think",
  "allenai/olmo-3-32b-think": "olmo-3-32b-think",
  "openrouter:amazon/nova-lite-v1": "nova-lite",
  "amazon/nova-lite-v1": "nova-lite",
  "openrouter:amazon/nova-micro-v1": "nova-micro",
  "amazon/nova-micro-v1": "nova-micro",
  "openrouter:amazon/nova-premier-v1": "nova-premier",
  "amazon/nova-premier-v1": "nova-premier",
  "openrouter:amazon/nova-pro-v1": "nova-pro",
  "amazon/nova-pro-v1": "nova-pro",
  "openrouter:anthracite-org/magnum-v4-72b": "magnum-v4-72b",
  "anthracite-org/magnum-v4-72b": "magnum-v4-72b",
  "openrouter:anthropic/claude-opus-4.1": "claude-opus-4.1",
  "anthropic/claude-opus-4.1": "claude-opus-4.1",
  "openrouter:anthropic/claude-opus-4.6-fast": "claude-opus-4.6-fast",
  "anthropic/claude-opus-4.6-fast": "claude-opus-4.6-fast",
  "openrouter:anthropic/claude-opus-4.7-fast": "claude-opus-4.7-fast",
  "anthropic/claude-opus-4.7-fast": "claude-opus-4.7-fast",
  "openrouter:anthropic/claude-opus-4.8-fast": "claude-opus-4.8-fast",
  "anthropic/claude-opus-4.8-fast": "claude-opus-4.8-fast",
  "openrouter:arcee-ai/coder-large": "coder-large",
  "arcee-ai/coder-large": "coder-large",
  "openrouter:arcee-ai/trinity-mini": "trinity-mini",
  "arcee-ai/trinity-mini": "trinity-mini",
  "openrouter:arcee-ai/virtuoso-large": "virtuoso-large",
  "arcee-ai/virtuoso-large": "virtuoso-large",
  "openrouter:baidu/ernie-4.5-vl-424b-a47b": "ernie-4.5-vl-424b-a47b",
  "baidu/ernie-4.5-vl-424b-a47b": "ernie-4.5-vl-424b-a47b",
  "openrouter:bytedance-seed/seed-1.6": "seed-1.6",
  "bytedance-seed/seed-1.6": "seed-1.6",
  "openrouter:bytedance-seed/seed-1.6-flash": "seed-1.6-flash",
  "bytedance-seed/seed-1.6-flash": "seed-1.6-flash",
  "openrouter:bytedance-seed/seed-2.0-lite": "seed-2.0-lite",
  "bytedance-seed/seed-2.0-lite": "seed-2.0-lite",
  "openrouter:bytedance-seed/seed-2.0-mini": "seed-2.0-mini",
  "bytedance-seed/seed-2.0-mini": "seed-2.0-mini",
  "openrouter:bytedance/ui-tars-1.5-7b": "ui-tars-1.5-7b",
  "ByteDance-Seed/UI-TARS-1.5-7B": "ui-tars-1.5-7b",
  "bytedance/ui-tars-1.5-7b": "ui-tars-1.5-7b",
  "openrouter:cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "dolphin-mistral-24b-venice-edition",
  "dphn/Dolphin-Mistral-24B-Venice-Edition": "dolphin-mistral-24b-venice-edition",
  "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "dolphin-mistral-24b-venice-edition",
  "openrouter:cohere/command-r-08-2024": "command-r24",
  "CohereLabs/c4ai-command-r-08-2024": "command-r24",
  "cohere/command-r-08-2024": "command-r24",
  "command-r7b-12-2024": "command-r7b24",
  "CohereLabs/c4ai-command-r7b-12-2024": "command-r7b24",
  "cohere/command-r7b-12-2024": "command-r7b24",
  "openrouter:cohere/north-mini-code:free": "north-mini-code",
  "cohere/north-mini-code:free": "north-mini-code",
  "openrouter:deepcogito/cogito-v2.1-671b": "cogito-v2.1-671b",
  "deepcogito/cogito-v2.1-671b": "cogito-v2.1-671b",
  "openrouter:deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3-0324",
  "deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3-0324",
  "openrouter:deepseek/deepseek-chat-v3.1": "deepseek-chat-v3.1",
  "deepseek/deepseek-chat-v3.1": "deepseek-chat-v3.1",
  "openrouter:deepseek/deepseek-r1-0528": "deepseek-r1-0528",
  "deepseek-ai/DeepSeek-R1-0528": "deepseek-r1-0528",
  "deepseek/deepseek-r1-0528": "deepseek-r1-0528",
  "openrouter:deepseek/deepseek-v3.1-terminus": "deepseek-v3.1-terminus",
  "deepseek-ai/DeepSeek-V3.1-Terminus": "deepseek-v3.1-terminus",
  "deepseek/deepseek-v3.1-terminus": "deepseek-v3.1-terminus",
  "openrouter:deepseek/deepseek-v3.2-exp": "deepseek-v3.2-exp",
  "deepseek-ai/DeepSeek-V3.2-Exp": "deepseek-v3.2-exp",
  "deepseek/deepseek-v3.2-exp": "deepseek-v3.2-exp",
  "openrouter:google/gemini-2.5-flash-image": "gemini-2.5-flash-image",
  "google/gemini-2.5-flash-image": "gemini-2.5-flash-image",
  "openrouter:google/gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite-preview25",
  "google/gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite-preview25",
  "openrouter:google/gemini-3-pro-image-preview": "gemini-3-pro-image",
  "google/gemini-3-pro-image-preview": "gemini-3-pro-image",
  "openrouter:google/gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
  "google/gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
  "openrouter:google/gemini-3.1-pro-preview-customtools": "gemini-3.1-pro-preview-customtools",
  "google/gemini-3.1-pro-preview-customtools": "gemini-3.1-pro-preview-customtools",
  "hf:google/gemma-2-27b-it": "gemma-2-27b-it",
  "openrouter:google/gemma-3-12b-it": "gemma-3-12b-it",
  "google/gemma-3-12b-it": "gemma-3-12b-it",
  "openrouter:google/gemma-3-4b-it": "gemma-3-4b-it",
  "google/gemma-3-4b-it": "gemma-3-4b-it",
  "openrouter:google/lyria-3-clip-preview": "lyria-3-clip",
  "google/lyria-3-clip-preview": "lyria-3-clip",
  "openrouter:google/lyria-3-pro-preview": "lyria-3-pro",
  "google/lyria-3-pro-preview": "lyria-3-pro",
  "openrouter:gryphe/mythomax-l2-13b": "mythomax-l2-13b",
  "gryphe/mythomax-l2-13b": "mythomax-l2-13b",
  "openrouter:ibm-granite/granite-4.0-h-micro": "granite-4.0-h-micro",
  "ibm-granite/granite-4.0-h-micro": "granite-4.0-h-micro",
  "openrouter:inclusionai/ling-2.6-1t": "ling-2.6-1t",
  "inclusionAI/Ling-2.6-1T": "ling-2.6-1t",
  "inclusionai/ling-2.6-1t": "ling-2.6-1t",
  "openrouter:inclusionai/ling-2.6-flash": "ling-2.6-flash",
  "inclusionai/ling-2.6-flash": "ling-2.6-flash",
  "openrouter:inclusionai/ring-2.6-1t": "ring-2.6-1t",
  "inclusionai/ring-2.6-1t": "ring-2.6-1t",
  "openrouter:inflection/inflection-3-pi": "inflection-3-pi",
  "inflection/inflection-3-pi": "inflection-3-pi",
  "openrouter:inflection/inflection-3-productivity": "inflection-3-productivity",
  "inflection/inflection-3-productivity": "inflection-3-productivity",
  "openrouter:kwaipilot/kat-coder-pro-v2": "kat-coder-pro",
  "kwaipilot/kat-coder-pro-v2": "kat-coder-pro",
  "openrouter:liquid/lfm-2-24b-a2b": "lfm-2-24b-a2b",
  "liquid/lfm-2-24b-a2b": "lfm-2-24b-a2b",
  "openrouter:liquid/lfm-2.5-1.2b-instruct:free": "lfm-2.5-1.2b",
  "liquid/lfm-2.5-1.2b-instruct:free": "lfm-2.5-1.2b",
  "openrouter:liquid/lfm-2.5-1.2b-thinking:free": "lfm-2.5-1.2b-thinking",
  "liquid/lfm-2.5-1.2b-thinking:free": "lfm-2.5-1.2b-thinking",
  "openrouter:mancer/weaver": "weaver",
  "mancer/weaver": "weaver",
  "openrouter:microsoft/phi-4": "phi-4",
  "microsoft/phi-4": "phi-4",
  "phi-4:14b": "phi-4",
  "openrouter:microsoft/phi-4-mini-instruct": "phi-4-mini",
  "microsoft/Phi-4-mini-instruct": "phi-4-mini",
  "microsoft/phi-4-mini-instruct": "phi-4-mini",
  "openrouter:microsoft/wizardlm-2-8x22b": "wizardlm-2-8x22b",
  "alpindale/WizardLM-2-8x22B": "wizardlm-2-8x22b",
  "microsoft/wizardlm-2-8x22b": "wizardlm-2-8x22b",
  "minimax/minimax-01": "minimax-01",
  "openrouter:minimax/minimax-m2-her": "minimax-m2-her",
  "minimax/minimax-m2-her": "minimax-m2-her",
  "openrouter:mistralai/mistral-large-2407": "mistral-large-2407",
  "mistralai/mistral-large-2407": "mistral-large-2407",
  "openrouter:mistralai/mistral-medium-3.1": "mistral-medium-3.1",
  "mistralai/mistral-medium-3.1": "mistral-medium-3.1",
  "openrouter:mistralai/mistral-saba": "mistral-saba",
  "mistralai/mistral-saba": "mistral-saba",
  "openrouter:mistralai/mistral-small-24b-instruct-2501": "mistral-small-24b-2501",
  "mistralai/mistral-small-24b-instruct-2501": "mistral-small-24b-2501",
  "openrouter:mistralai/mistral-small-3.2-24b-instruct": "mistral-small-3.2-24b",
  "mistralai/mistral-small-3.2-24b-instruct": "mistral-small-3.2-24b",
  "open-mixtral-8x22b": "mixtral-8x22b",
  "mistralai/mixtral-8x22b-instruct": "mixtral-8x22b",
  "openrouter:mistralai/voxtral-small-24b-2507": "voxtral-small-24b-2507",
  "mistralai/voxtral-small-24b-2507": "voxtral-small-24b-2507",
  "openrouter:moonshotai/kimi-k2-thinking": "kimi-k2-thinking",
  "moonshotai/Kimi-K2-Thinking": "kimi-k2-thinking",
  "kimi-k2-thinking:cloud": "kimi-k2-thinking",
  "moonshotai/kimi-k2-thinking": "kimi-k2-thinking",
  "openrouter:morph/morph-v3-fast": "morph-v3-fast",
  "morph/morph-v3-fast": "morph-v3-fast",
  "openrouter:morph/morph-v3-large": "morph-v3-large",
  "morph/morph-v3-large": "morph-v3-large",
  "openrouter:nex-agi/nex-n2-pro": "nex-n2-pro",
  "nex-agi/nex-n2-pro": "nex-n2-pro",
  "openrouter:nousresearch/hermes-3-llama-3.1-405b:free": "hermes-3-llama-3.1-405b",
  "nousresearch/hermes-3-llama-3.1-405b": "hermes-3-llama-3.1-405b",
  "openrouter:nousresearch/hermes-3-llama-3.1-70b": "hermes-3-70b",
  "nousresearch/hermes-3-llama-3.1-70b": "hermes-3-llama-3.1-70b",
  "openrouter:nousresearch/hermes-4-405b": "hermes-4-405b",
  "nousresearch/hermes-4-405b": "hermes-4-405b",
  "openrouter:nousresearch/hermes-4-70b": "hermes-4-70b",
  "NousResearch/Hermes-4-70B": "hermes-4-70b",
  "nousresearch/hermes-4-70b": "hermes-4-70b",
  "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1.5": "llama-3.3-nemotron-super-49b-v1.5",
  "nvidia/llama-3.3-nemotron-super-49b-v1.5": "llama-3.3-nemotron-super-49b-v1.5",
  "openrouter:nvidia/nemotron-3-nano-30b-a3b:free": "nemotron-3-nano-30b-a3b",
  "nvidia/nemotron-3-nano-30b-a3b": "nemotron-3-nano-30b-a3b",
  "openrouter:nvidia/nemotron-3-super-120b-a12b:free": "nemotron-3-super-120b-a12b",
  "nvidia/nemotron-3-super-120b-a12b": "nemotron-3-super-120b-a12b",
  "togetherai:nvidia/nemotron-3-ultra-550b-a55b": "nemotron-3-ultra-550b-a55b",
  "nvidia/nemotron-3-ultra-550b-a55b": "nemotron-3-ultra-550b-a55b",
  "openrouter:nvidia/nemotron-3.5-content-safety:free": "nemotron-3.5-content-safety",
  "nvidia/nemotron-3.5-content-safety:free": "nemotron-3.5-content-safety",
  "openrouter:nvidia/nemotron-nano-12b-v2-vl:free": "nemotron-nano-12b-v2-vl",
  "nvidia/nemotron-nano-12b-v2-vl:free": "nemotron-nano-12b-v2-vl",
  "openrouter:nvidia/nemotron-nano-9b-v2:free": "nemotron-nano-9b",
  "nvidia/nemotron-nano-9b-v2:free": "nemotron-nano-9b",
  "openrouter:openai/gpt-3.5-turbo-instruct": "gpt-3.5-turbo",
  "openai/gpt-3.5-turbo": "gpt-3.5-turbo",
  "openrouter:openai/gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
  "openai/gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
  "openrouter:openai/gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
  "openai/gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
  "openrouter:openai/gpt-4-turbo-preview": "gpt-4-turbo",
  "openai/gpt-4-turbo-preview": "gpt-4-turbo",
  "openrouter:openai/gpt-4o-mini-search-preview": "gpt-4o-mini-search",
  "openai/gpt-4o-mini-search-preview": "gpt-4o-mini-search",
  "openrouter:openai/gpt-4o-search-preview": "gpt-4o-search",
  "openai/gpt-4o-search-preview": "gpt-4o-search",
  "openrouter:openai/gpt-5-image": "gpt-5-image",
  "openai/gpt-5-image": "gpt-5-image",
  "openrouter:openai/gpt-5-image-mini": "gpt-5-image-mini",
  "openai/gpt-5-image-mini": "gpt-5-image-mini",
  "openrouter:openai/gpt-5-pro": "gpt-5-pro",
  "openai/gpt-5-pro": "gpt-5-pro",
  "openrouter:openai/gpt-5.1-codex-max": "gpt-5.1-codex-max",
  "openai/gpt-5.1-codex-max": "gpt-5.1-codex-max",
  "openrouter:openai/gpt-5.4-image-2": "gpt-5.4-image-2",
  "openai/gpt-5.4-image-2": "gpt-5.4-image-2",
  "openrouter:openai/gpt-audio": "gpt-audio",
  "openai/gpt-audio": "gpt-audio",
  "openrouter:openai/gpt-audio-mini": "gpt-audio-mini",
  "openai/gpt-audio-mini": "gpt-audio-mini",
  "openrouter:openai/gpt-chat-latest": "gpt-chat",
  "openai/gpt-chat-latest": "gpt-chat",
  "openrouter:openai/gpt-oss-safeguard-20b": "gpt-oss-safeguard-20b",
  "openai/gpt-oss-safeguard-20b": "gpt-oss-safeguard-20b",
  "openrouter:openai/o3-deep-research": "o3-deep-research",
  "openai/o3-deep-research": "o3-deep-research",
  "openrouter:openai/o4-mini-deep-research": "o4-mini-deep-research",
  "openai/o4-mini-deep-research": "o4-mini-deep-research",
  "openrouter:openrouter/bodybuilder": "bodybuilder",
  "openrouter/bodybuilder": "bodybuilder",
  "openrouter:openrouter/free": "free",
  "openrouter/free": "free",
  "openrouter:openrouter/owl-alpha": "owl-alpha",
  "openrouter/owl-alpha": "owl-alpha",
  "openrouter:openrouter/pareto-code": "pareto-code",
  "openrouter/pareto-code": "pareto-code",
  "openrouter:perceptron/perceptron-mk1": "perceptron-mk1",
  "perceptron/perceptron-mk1": "perceptron-mk1",
  "openrouter:perplexity/sonar-deep-research": "sonar-deep-research",
  "perplexity/sonar-deep-research": "sonar-deep-research",
  "openrouter:perplexity/sonar-pro-search": "sonar-pro-search",
  "perplexity/sonar-pro-search": "sonar-pro-search",
  "openrouter:poolside/laguna-m.1:free": "laguna-m.1",
  "poolside/laguna-m.1": "laguna-m.1",
  "openrouter:poolside/laguna-xs.2:free": "laguna-xs.2",
  "poolside/laguna-xs.2": "laguna-xs.2",
  "openrouter:qwen/qwen3-30b-a3b-thinking-2507": "qwen-3-30b-a3b-thinking-2507",
  "qwen/qwen3-30b-a3b-thinking-2507": "qwen-3-30b-a3b-thinking-2507",
  "openrouter:qwen/qwen3-coder:free": "qwen-3-coder",
  "qwen3-coder": "qwen-3-coder",
  "qwen3-coder:480b": "qwen-3-coder",
  "qwen3-coder:30b": "qwen-3-coder",
  "qwen/qwen3-coder": "qwen-3-coder",
  "openrouter:qwen/qwen3-vl-30b-a3b-thinking": "qwen-3-vl-30b-a3b-thinking",
  "Qwen/Qwen3-VL-30B-A3B-Thinking": "qwen-3-vl-30b-a3b-thinking",
  "qwen/qwen3-vl-30b-a3b-thinking": "qwen-3-vl-30b-a3b-thinking",
  "openrouter:qwen/qwen3-vl-32b-instruct": "qwen-3-vl-32b",
  "qwen3-vl-32b": "qwen-3-vl-32b",
  "qwen/qwen3-vl-32b-instruct": "qwen-3-vl-32b",
  "openrouter:qwen/qwen3.6-flash": "qwen-3.6-flash",
  "qwen/qwen3.6-flash": "qwen-3.6-flash",
  "openrouter:rekaai/reka-edge": "reka-edge",
  "rekaai/reka-edge": "reka-edge",
  "openrouter:rekaai/reka-flash-3": "reka-flash-3",
  "rekaai/reka-flash-3": "reka-flash-3",
  "openrouter:relace/relace-apply-3": "relace-apply-3",
  "relace/relace-apply-3": "relace-apply-3",
  "openrouter:relace/relace-search": "relace-search",
  "relace/relace-search": "relace-search",
  "openrouter:sakana/fugu-ultra": "fugu-ultra",
  "sakana/fugu-ultra": "fugu-ultra",
  "openrouter:sao10k/l3-lunaris-8b": "l3-lunaris-8b",
  "sao10k/l3-lunaris-8b": "l3-lunaris-8b",
  "openrouter:sao10k/l3.1-70b-hanami-x1": "l3.1-70b-hanami-x1",
  "sao10k/l3.1-70b-hanami-x1": "l3.1-70b-hanami-x1",
  "openrouter:sao10k/l3.1-euryale-70b": "l3.1-euryale-70b",
  "sao10k/l3.1-euryale-70b": "l3.1-euryale-70b",
  "openrouter:sao10k/l3.3-euryale-70b": "l3.3-euryale-70b",
  "sao10k/l3.3-euryale-70b": "l3.3-euryale-70b",
  "openrouter:switchpoint/router": "router",
  "switchpoint/router": "router",
  "openrouter:tencent/hunyuan-a13b-instruct": "hunyuan-a13b",
  "tencent/hunyuan-a13b-instruct": "hunyuan-a13b",
  "openrouter:tencent/hy3-preview": "hy3",
  "tencent/hy3-preview": "hy3",
  "openrouter:thedrummer/cydonia-24b-v4.1": "cydonia-24b-v4.1",
  "thedrummer/cydonia-24b-v4.1": "cydonia-24b-v4.1",
  "openrouter:thedrummer/rocinante-12b": "rocinante-12b",
  "thedrummer/rocinante-12b": "rocinante-12b",
  "openrouter:thedrummer/skyfall-36b-v2": "skyfall-36b",
  "thedrummer/skyfall-36b-v2": "skyfall-36b",
  "openrouter:thedrummer/unslopnemo-12b": "unslopnemo-12b",
  "thedrummer/unslopnemo-12b": "unslopnemo-12b",
  "openrouter:undi95/remm-slerp-l2-13b": "remm-slerp-l2-13b",
  "undi95/remm-slerp-l2-13b": "remm-slerp-l2-13b",
  "openrouter:upstage/solar-pro-3": "solar-pro-3",
  "upstage/solar-pro-3": "solar-pro-3",
  "openrouter:writer/palmyra-x5": "palmyra-x5",
  "writer/palmyra-x5": "palmyra-x5",
  "openrouter:x-ai/grok-4.20": "grok-4.20",
  "x-ai/grok-4.20": "grok-4.20",
  "openrouter:x-ai/grok-4.20-multi-agent": "grok-4.20-multi-agent",
  "x-ai/grok-4.20-multi-agent": "grok-4.20-multi-agent",
  "openrouter:x-ai/grok-build-0.1": "grok-build-0.1",
  "x-ai/grok-build-0.1": "grok-build-0.1",
  "openrouter:~google/gemini-flash-latest": "gemini-flash",
  "~google/gemini-flash-latest": "gemini-flash",
  "openrouter:~google/gemini-pro-latest": "gemini-pro",
  "~google/gemini-pro-latest": "gemini-pro",
  "openrouter:~openai/gpt-latest": "gpt",
  "~openai/gpt-latest": "gpt",
  "openrouter:~openai/gpt-mini-latest": "gpt-mini",
  "~openai/gpt-mini-latest": "gpt-mini",
  "togetherai:arize-ai/qwen-2-1.5b-instruct": "qwen-2-1.5b",
  "togetherai:deepcogito/cogito-v2-1-671b": "cogito-v2-1-671b",
  "togetherai:liquidai/lfm2-24b-a2b": "lfm2-24b-a2b",
  "togetherai:meta-llama/llama-3.3-70b-instruct-turbo": "llama-3.3-70b-turbo",
  "togetherai:meta-llama/llama-guard-4-12b": "llama-guard-4-12b",
  "meta-llama/Llama-Guard-4-12B": "llama-guard-4-12b",
  "meta-llama/llama-guard-4-12b": "llama-guard-4-12b",
  "togetherai:meta-llama/meta-llama-3-8b-instruct-lite": "llama-3-8b-lite",
  "togetherai:qwen/qwen2.5-7b-instruct-turbo": "qwen-2.5-7b-turbo",
  "togetherai:qwen/qwen3.5-9b": "qwen-3.5-9b",
  "Qwen/Qwen3.5-9B": "qwen-3.5-9b",
  "qwen/qwen3.5-9b": "qwen-3.5-9b",
  "x-ai:x-ai/grok-2-vision": "grok-2-vision",
  "x-ai:x-ai/grok-2-vision-1212": "grok-2-vision-1212",
  "x-ai:x-ai/grok-3-fast": "grok-3-fast",
  "x-ai:x-ai/grok-3-mini-fast": "grok-3-mini-fast",
  "x-ai:x-ai/grok-4-0709": "grok-4-0709",
  "x-ai:x-ai/grok-4-1-fast": "grok-4-1-fast",
  "x-ai:x-ai/grok-4-fast": "grok-4-fast",
  "x-ai:x-ai/grok-4-fast-non-reasoning": "grok-4-fast-non-reasoning",
  "x-ai:x-ai/grok-code-fast-1": "grok-code-fast-1",
  "x-ai:x-ai/grok-vision-beta": "grok-vision-beta",
  "z-ai:z-ai/autoglm-phone-multilingual": "autoglm-phone-multilingual",
  "z-ai:z-ai/glm-4-32b-0414-128k": "glm-4-32b-0414-128k",
  "z-ai:z-ai/glm-4.5": "glm-4.5",
  "zai-org/GLM-4.5": "glm-4.5",
  "z-ai/glm-4.5": "glm-4.5",
  "z-ai:z-ai/glm-4.5-air": "glm-4.5-air",
  "zai-org/GLM-4.5-Air": "glm-4.5-air",
  "z-ai/glm-4.5-air": "glm-4.5-air",
  "z-ai:z-ai/glm-4.5-airx": "glm-4.5-airx",
  "z-ai:z-ai/glm-4.5-flash": "glm-4.5-flash",
  "z-ai:z-ai/glm-4.5-x": "glm-4.5-x",
  "z-ai:z-ai/glm-4.5v": "glm-4.5v",
  "zai-org/GLM-4.5V-FP8": "glm-4.5v",
  "zai-org/GLM-4.5V": "glm-4.5v",
  "z-ai/glm-4.5v": "glm-4.5v",
  "z-ai:z-ai/glm-4.6": "glm-4.6",
  "zai-org/GLM-4.6": "glm-4.6",
  "glm-4.6:cloud": "glm-4.6",
  "z-ai/glm-4.6": "glm-4.6",
  "z-ai:z-ai/glm-4.6v": "glm-4.6v",
  "zai-org/GLM-4.6V-FP8": "glm-4.6v",
  "z-ai/glm-4.6v": "glm-4.6v",
  "z-ai:z-ai/glm-4.6v-flash": "glm-4.6v-flash",
  "zai-org/GLM-4.6V-Flash": "glm-4.6v-flash",
  "z-ai:z-ai/glm-4.6v-flashx": "glm-4.6v-flashx",
  "z-ai:z-ai/glm-4.7-flashx": "glm-4.7-flashx",
  "z-ai:z-ai/glm-5-turbo": "glm-5-turbo",
  "z-ai/glm-5-turbo": "glm-5-turbo",
  "pixtral-large-latest": "pixtral-large",
  "openrouter:meta-llama/llama-3.3-8b-instruct:free": "llama-3.3-8b",
  "openrouter:google/gemini-flash-1.5-8b": "gemini-1.5-8b-flash",
  "openrouter:google/gemini-pro-1.5": "gemini-1.5-pro",
  "openrouter:google/gemini-2.5-flash-preview:thinking": "gemini-2.5-flash-thinking",
  "openrouter:google/gemma-3-1b-it:free": "gemma-3-1b",
  "openrouter:nousresearch/hermes-2-pro-llama-3-8b": "hermes-2-pro",
  "openrouter:nousresearch/hermes-3-llama-3.1-405b": "hermes-3-405b",
  "openrouter:nousresearch/deephermes-3-llama-3-8b-preview:free": "deephermes-3-8b",
  "openrouter:nousresearch/deephermes-3-mistral-24b-preview:free": "deephermes-3-24b",
  "openrouter:microsoft/phi-3-mini-128k-instruct": "phi-3-mini",
  "openrouter:microsoft/phi-3-medium-128k-instruct": "phi-3-medium",
  "openrouter:microsoft/phi-4-multimodal-instruct": "phi-4-multimodal",
  "openrouter:microsoft/phi-4-reasoning:free": "phi-4-reasoning",
  "openrouter:microsoft/mai-ds-r1:free": "mai-ds-r1",
  "openrouter:anthropic/claude-3.7-sonnet:thinking": "claude-3.7-sonnet-thinking",
  "claude-3-opus-latest": "claude-3-opus",
  "claude-3-sonnet-20240229": "claude-3-sonnet",
  "openrouter:rekaai/reka-flash-3:free": "reka-flash",
  "openrouter:cohere/command": "command",
  "openrouter:qwen/qwen3-0.6b-04-28:free": "qwen-3-0.6b",
  "qwen3-0.6b": "qwen-3-0.6b",
  "Qwen/Qwen3-0.6B": "qwen-3-0.6b",
  "openrouter:qwen/qwen3-1.7b:free": "qwen-3-1.7b",
  "qwen3-1.7b": "qwen-3-1.7b",
  "Qwen/Qwen3-1.7B": "qwen-3-1.7b",
  "openrouter:qwen/qwen3-4b:free": "qwen-3-4b",
  "qwen3-4b": "qwen-3-4b",
  "Qwen/Qwen3-4B": "qwen-3-4b",
  "openrouter:qwen/qwen2.5-coder-7b-instruct": "qwen-2.5-coder-7b",
  "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-7b",
  "Qwen/Qwen2.5-Coder-7B": "qwen-2.5-coder-7b",
  "openrouter:qwen/qwen2.5-vl-3b-instruct:free": "qwen-2.5-vl-3b",
  "deepseek-ai/DeepSeek-V3-0324": "deepseek-v3-0324",
  "lordoliver/DeepSeek-V3-0324:671b-q4_k_m": "deepseek-v3-0324",
  "openrouter:deepseek/deepseek-r1-zero:free": "deepseek-r1-zero",
  "openrouter:deepseek/deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill-llama-8b",
  "deepseek/deepseek-chat": "deepseek-chat",
  "deepseek-coder:6.7b": "deepseek-coder",
  "openrouter:x-ai/grok-3-beta": "grok-3-beta",
  "openrouter:perplexity/llama-3.1-sonar-small-128k-online": "llama-3.1-sonar-small-online",
  "openrouter:perplexity/llama-3.1-sonar-large-128k-online": "llama-3.1-sonar-large-online",
  "openrouter:nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "nemotron-253b",
  "openrouter:thudm/glm-4-9b:free": "glm-4-9b",
  "openrouter:thudm/glm-z1-9b:free": "glm-z1-9b",
  "openrouter:thudm/glm-z1-rumination-32b": "glm-z1-rumination-32b",
  "openrouter:cognitivecomputations/dolphin3.0-r1-mistral-24b:free": "dolphin-3.0-r1-24b",
  "openrouter:cognitivecomputations/dolphin3.0-mistral-24b:free": "dolphin-3.0-24b",
  "openrouter:cognitivecomputations/dolphin-mixtral-8x22b": "dolphin-8x22b",
  "openrouter:agentica-org/deepcoder-14b-preview:free": "deepcoder-14b",
  "openrouter:moonshotai/kimi-vl-a3b-thinking:free": "kimi-vl-thinking",
  "openrouter:moonshotai/moonlight-16b-a3b-instruct:free": "moonlight-16b",
  "openrouter:featherless/qwerky-72b:free": "qwerky-72b",
  "openrouter:liquid/lfm-7b": "lfm-7b",
  "openrouter:liquid/lfm-3b": "lfm-3b",
  "openrouter:liquid/lfm-40b": "lfm-40b",
  "CohereLabs/c4ai-command-a-03-2025": "command-a25",
  "command-r7b-arabic-02-2025": "command-r7b-arabic25",
  "CohereLabs/c4ai-command-r7b-arabic-02-2025": "command-r7b-arabic25"
}
