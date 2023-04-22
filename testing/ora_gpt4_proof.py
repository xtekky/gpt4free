import ora

complex_question = '''
James is talking to two people, his father, and his friend. 

Douglas asks him, "What did you do today James?" 
James replies, "I went on a fishing trip." 
Josh then asks, "Did you catch anything?" 
James replies, "Yes, I caught a couple of nice rainbow trout. It was a lot of fun." 
Josh replies, "Good job son, tell your mother we should eat them tonight, she'll be very happy." 
Douglas then says, "I wish my family would eat fish tonight, my father is making pancakes." 

Question: Who is James' father? 
'''

# right answer is josh

model = ora.CompletionModel.load('b8b12eaa-5d47-44d3-92a6-4d706f2bcacf', 'gpt-4')
# init conversation (will give you a conversationId)
init = ora.Completion.create(
    model  = model,
    prompt = complex_question)

print(init.completion.choices[0].text) # James' father is Josh.