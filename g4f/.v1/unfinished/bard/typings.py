from typing import Dict, List, Union


class BardResponse:
    def __init__(self, json_dict: Dict[str, Union[str, List]]) -> None:
        """
        Initialize a BardResponse object.

        :param json_dict: A dictionary containing the JSON response data.
        """
        self.json = json_dict

        self.content = json_dict.get('content')
        self.conversation_id = json_dict.get('conversation_id')
        self.response_id = json_dict.get('response_id')
        self.factuality_queries = json_dict.get('factualityQueries', [])
        self.text_query = json_dict.get('textQuery', [])
        self.choices = [self.BardChoice(choice)
                        for choice in json_dict.get('choices', [])]

    def __repr__(self) -> str:
        """
        Return a string representation of the BardResponse object.

        :return: A string representation of the BardResponse object.
        """
        return f"BardResponse(conversation_id={self.conversation_id}, response_id={self.response_id}, content={self.content})"

    def filter_choices(self, keyword: str) -> List['BardChoice']:
        """
        Filter the choices based on a keyword.

        :param keyword: The keyword to filter choices by.
        :return: A list of filtered BardChoice objects.
        """
        return [choice for choice in self.choices if keyword.lower() in choice.content.lower()]

    class BardChoice:
        def __init__(self, choice_dict: Dict[str, str]) -> None:
            """
            Initialize a BardChoice object.

            :param choice_dict: A dictionary containing the choice data.
            """
            self.id = choice_dict.get('id')
            self.content = choice_dict.get('content')[0]

        def __repr__(self) -> str:
            """
            Return a string representation of the BardChoice object.

            :return: A string representation of the BardChoice object.
            """
            return f"BardChoice(id={self.id}, content={self.content})"
