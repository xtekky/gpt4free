from __future__ import annotations

from aiohttp import ClientSession
import time
import asyncio

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse

class Prodia(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://app.prodia.com"
    api_endpoint = "https://api.prodia.com/generate"
    working = True
    
    default_model = 'absolutereality_v181.safetensors [3d9d4d2b]'
    models = [
        '3Guofeng3_v34.safetensors [50f420de]',
        'absolutereality_V16.safetensors [37db0fc3]',
        'absolutereality_v181.safetensors [3d9d4d2b]',
        'amIReal_V41.safetensors [0a8a2e61]',
        'analog-diffusion-1.0.ckpt [9ca13f02]',
        'aniverse_v30.safetensors [579e6f85]',
        'anythingv3_0-pruned.ckpt [2700c435]',
        'anything-v4.5-pruned.ckpt [65745d25]',
        'anythingV5_PrtRE.safetensors [893e49b9]',
        'AOM3A3_orangemixs.safetensors [9600da17]',
        'blazing_drive_v10g.safetensors [ca1c1eab]',
        'breakdomain_I2428.safetensors [43cc7d2f]',
        'breakdomain_M2150.safetensors [15f7afca]',
        'cetusMix_Version35.safetensors [de2f2560]',
        'childrensStories_v13D.safetensors [9dfaabcb]',
        'childrensStories_v1SemiReal.safetensors [a1c56dbb]',
        'childrensStories_v1ToonAnime.safetensors [2ec7b88b]',
        'Counterfeit_v30.safetensors [9e2a8f19]',
        'cuteyukimixAdorable_midchapter3.safetensors [04bdffe6]',
        'cyberrealistic_v33.safetensors [82b0d085]',
        'dalcefo_v4.safetensors [425952fe]',
        'deliberate_v2.safetensors [10ec4b29]',
        'deliberate_v3.safetensors [afd9d2d4]',
        'dreamlike-anime-1.0.safetensors [4520e090]',
        'dreamlike-diffusion-1.0.safetensors [5c9fd6e0]',
        'dreamlike-photoreal-2.0.safetensors [fdcf65e7]',
        'dreamshaper_6BakedVae.safetensors [114c8abb]',
        'dreamshaper_7.safetensors [5cf5ae06]',
        'dreamshaper_8.safetensors [9d40847d]',
        'edgeOfRealism_eorV20.safetensors [3ed5de15]',
        'EimisAnimeDiffusion_V1.ckpt [4f828a15]',
        'elldreths-vivid-mix.safetensors [342d9d26]',
        'epicphotogasm_xPlusPlus.safetensors [1a8f6d35]',
        'epicrealism_naturalSinRC1VAE.safetensors [90a4c676]',
        'epicrealism_pureEvolutionV3.safetensors [42c8440c]',
        'ICantBelieveItsNotPhotography_seco.safetensors [4e7a3dfd]',
        'indigoFurryMix_v75Hybrid.safetensors [91208cbb]',
        'juggernaut_aftermath.safetensors [5e20c455]',
        'lofi_v4.safetensors [ccc204d6]',
        'lyriel_v16.safetensors [68fceea2]',
        'majicmixRealistic_v4.safetensors [29d0de58]',
        'mechamix_v10.safetensors [ee685731]',
        'meinamix_meinaV9.safetensors [2ec66ab0]',
        'meinamix_meinaV11.safetensors [b56ce717]',
        'neverendingDream_v122.safetensors [f964ceeb]',
        'openjourney_V4.ckpt [ca2f377f]',
        'pastelMixStylizedAnime_pruned_fp16.safetensors [793a26e8]',
        'portraitplus_V1.0.safetensors [1400e684]',
        'protogenx34.safetensors [5896f8d5]',
        'Realistic_Vision_V1.4-pruned-fp16.safetensors [8d21810b]',
        'Realistic_Vision_V2.0.safetensors [79587710]',
        'Realistic_Vision_V4.0.safetensors [29a7afaa]',
        'Realistic_Vision_V5.0.safetensors [614d1063]',
        'Realistic_Vision_V5.1.safetensors [a0f13c83]',
        'redshift_diffusion-V10.safetensors [1400e684]',
        'revAnimated_v122.safetensors [3f4fefd9]',
        'rundiffusionFX25D_v10.safetensors [cd12b0ee]',
        'rundiffusionFX_v10.safetensors [cd4e694d]',
        'sdv1_4.ckpt [7460a6fa]',
        'v1-5-pruned-emaonly.safetensors [d7049739]',
        'v1-5-inpainting.safetensors [21c7ab71]',
        'shoninsBeautiful_v10.safetensors [25d8c546]',
        'theallys-mix-ii-churned.safetensors [5d9225a4]',
        'timeless-1.0.ckpt [7c4971d4]',
        'toonyou_beta6.safetensors [980f6b15]',
    ]

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]['content'] if messages else ""
            
            params = {
                "new": "true",
                "prompt": prompt,
                "model": model,
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "steps": kwargs.get("steps", 20),
                "cfg": kwargs.get("cfg", 7),
                "seed": kwargs.get("seed", int(time.time())),
                "sampler": kwargs.get("sampler", "DPM++ 2M Karras"),
                "aspect_ratio": kwargs.get("aspect_ratio", "square")
            }
            
            async with session.get(cls.api_endpoint, params=params, proxy=proxy) as response:
                response.raise_for_status()
                job_data = await response.json()
                job_id = job_data["job"]
                
                image_url = await cls._poll_job(session, job_id, proxy)
                yield ImageResponse(image_url, alt=prompt)

    @classmethod
    async def _poll_job(cls, session: ClientSession, job_id: str, proxy: str, max_attempts: int = 30, delay: int = 2) -> str:
        for _ in range(max_attempts):
            async with session.get(f"https://api.prodia.com/job/{job_id}", proxy=proxy) as response:
                response.raise_for_status()
                job_status = await response.json()

                if job_status["status"] == "succeeded":
                    return f"https://images.prodia.xyz/{job_id}.png"
                elif job_status["status"] == "failed":
                    raise Exception("Image generation failed")

            await asyncio.sleep(delay)

        raise Exception("Timeout waiting for image generation")
