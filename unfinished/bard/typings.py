class BardResponse:
    def __init__(self, json_dict):
        self.json               = json_dict
        
        self.content            = json_dict.get('content')
        self.conversation_id    = json_dict.get('conversation_id')
        self.response_id        = json_dict.get('response_id')
        self.factuality_queries = json_dict.get('factualityQueries', [])
        self.text_query         = json_dict.get('textQuery', [])
        self.choices            = [self.BardChoice(choice) for choice in json_dict.get('choices', [])]

    class BardChoice:
        def __init__(self, choice_dict):
            self.id      = choice_dict.get('id')
            self.content = choice_dict.get('content')[0]
