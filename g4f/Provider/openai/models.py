default_model = "auto"
default_image_model = "gpt-image"
image_models = [default_image_model]
text_models = [default_model, "gpt-5", "gpt-5-thinking", "gpt-4", "gpt-4.1", "gpt-4.1-mini", "gpt-4.5", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3-mini-high", "o4-mini", "o4-mini-high"]
vision_models = text_models
models = text_models + image_models
model_aliases = {
    "gpt-4.1": "gpt-4-1",
    "gpt-4.1-mini": "gpt-4-1-mini",
    "gpt-4.5": "gpt-4-5",
    "dall-e-3": default_image_model,
}