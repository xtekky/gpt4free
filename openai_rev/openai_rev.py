from enum import Enum

import quora
import you


class Provider(Enum):
    You = 'you'
    Poe = 'poe'


class Completion:
    @staticmethod
    def create(provider: Provider, prompt: str, **kwargs):
        if provider == Provider.Poe:
            return Completion.__poe_service(prompt, **kwargs)
        elif provider == Provider.You:
            return Completion.__you_service(prompt, **kwargs)

    @classmethod
    def __you_service(cls, prompt: str, **kwargs) -> str:
        return you.Completion.create(prompt).text

    @classmethod
    def __poe_service(cls, prompt: str, **kwargs) -> str:
        return quora.Completion.create(prompt=prompt).text


# usage You
response = Completion.create(Provider.You, prompt='Write a poem on Lionel Messi')
print(response)

# usage Poe
response = Completion.create(Provider.Poe, prompt='Write a poem on Lionel Messi', token='GKzCahZYGKhp76LfE197xw==')
print(response)
