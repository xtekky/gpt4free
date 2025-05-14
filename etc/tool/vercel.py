import json
import re
from typing import Any

import quickjs
from curl_cffi import requests

session = requests.Session(impersonate="chrome107")


def get_model_info() -> dict[str, Any]:
    url = "https://sdk.vercel.ai"
    response = session.get(url)
    html = response.text
    paths_regex = r"static\/chunks.+?\.js"
    separator_regex = r'"\]\)<\/script><script>self\.__next_f\.push\(\[.,"'

    paths = re.findall(paths_regex, html)
    paths = [re.sub(separator_regex, "", path) for path in paths]
    paths = list(set(paths))

    urls = [f"{url}/_next/{path}" for path in paths]
    scripts = [session.get(url).text for url in urls]

    models_regex = r'let .="\\n\\nHuman:\",r=(.+?),.='
    for script in scripts:

        matches = re.findall(models_regex, script)
        if matches:
            models_str = matches[0]
            stop_sequences_regex = r"(?<=stopSequences:{value:\[)\D(?<!\])"
            models_str = re.sub(
                stop_sequences_regex, re.escape('"\\n\\nHuman:"'), models_str
            )

            context = quickjs.Context()  # type: ignore
            json_str: str = context.eval(f"({models_str})").json()  # type: ignore
            return json.loads(json_str)  # type: ignore

    return {}


def convert_model_info(models: dict[str, Any]) -> dict[str, Any]:
    model_info: dict[str, Any] = {}
    for model_name, params in models.items():
        default_params = params_to_default_params(params["parameters"])
        model_info[model_name] = {"id": params["id"], "default_params": default_params}
    return model_info


def params_to_default_params(parameters: dict[str, Any]):
    defaults: dict[str, Any] = {}
    for key, parameter in parameters.items():
        if key == "maximumLength":
            key = "maxTokens"
        defaults[key] = parameter["value"]
    return defaults


def get_model_names(model_info: dict[str, Any]):
    model_names = model_info.keys()
    model_names = [
        name
        for name in model_names
        if name not in ["openai:gpt-4", "openai:gpt-3.5-turbo"]
    ]
    model_names.sort()
    return model_names


def print_providers(model_names: list[str]):
    for name in model_names:
        split_name = re.split(r":|/", name)
        base_provider = split_name[0]
        variable_name = split_name[-1].replace("-", "_").replace(".", "")
        line = f'{variable_name} = Model(name="{name}", base_provider="{base_provider}", best_provider=Vercel,)\n'
        print(line)


def print_convert(model_names: list[str]):
    for name in model_names:
        split_name = re.split(r":|/", name)
        key = split_name[-1]
        variable_name = split_name[-1].replace("-", "_").replace(".", "")
        # "claude-instant-v1": claude_instant_v1,
        line = f'        "{key}": {variable_name},'
        print(line)


def main():
    model_info = get_model_info()
    model_info = convert_model_info(model_info)
    print(json.dumps(model_info, indent=2))

    model_names = get_model_names(model_info)
    print("-------" * 40)
    print_providers(model_names)
    print("-------" * 40)
    print_convert(model_names)


if __name__ == "__main__":
    main()
