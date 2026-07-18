from ....config import DEFAULT_MODEL

default_image_model = "black-forest-labs/FLUX.1-dev"
image_models = [    
    default_image_model,
    "black-forest-labs/FLUX.1-schnell",
]
image_model_aliases = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "stable-diffusion-3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd-3.5-large": "stabilityai/stable-diffusion-3.5-large",
}
