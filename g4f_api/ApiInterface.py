import g4f
from g4f.api import Api

create_chat_completion_original = g4f.ChatCompletion.create
list_ignored_providers=[]

def create_chat_completion(*args, **kwargs):
	kwargs['ignored']=list_ignored_providers
	return create_chat_completion_original(*args, **kwargs)

g4f.ChatCompletion.create=create_chat_completion
api=Api(g4f, debug=False)
