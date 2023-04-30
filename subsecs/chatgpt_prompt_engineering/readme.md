---
layout: default
---

[back](../gpt.md)

# ChatGPT Prompt Engineering for Developers

## Contents

### Lesson 1: guidelines
[notes](./l2-guidelines%20-%20Jupyter%20Notebook.pdf) <br>
Prompting Principles <br>
- Principle 1: Write clear and specific instructions
- Principle 2: Give the model time to “think”

Principle 1: Write clear and specific instructions <br>
- Tactic 1: Use delimiters to clearly indicate distinct parts of the input
- Tactic 2: Ask for a structured output. E.g. JSON or HTML
- Tactic 3: Ask the model to check whether conditions are satisfied
- Tactic 4: "Few-shot" prompting

Principle 2: Give the model time to “think” <br>
- Tactic 1: Specify the steps required to complete a task
- Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion

### Lesson 2: iterative
[notes](./l3-iterative-prompt-development%20-%20Jupyter%20Notebook.pdf) <br>
iteratively analyze and refine your prompts to generate.

### Lesson 3: summarizing
[notes](./l4-summarizing%20-%20Jupyter%20Notebook.pdf) <br>
summarize text with a focus on specific topics.

### Lesson 4: inferring
[notes](./l5-inferring%20-%20Jupyter%20Notebook.pdf)
infer sentiment and topics from product reviews and news articles.

### Lesson 5: transforming
[notes](./l6-transforming%20-%20Jupyter%20Notebook.pdf) <br>
language translation, spelling and grammar checking, tone adjustment, and format conversion.

### Lesson 6: expanding
[notes](./l7-expanding%20-%20Jupyter%20Notebook.pdf) <br>
generate customer service emails that are tailored to each customer's review. <br>
```python
# degree of exploration / randomness
response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
```

### Lesson 7: chatbot
[notes](./l8-chatbot%20-%20Jupyter%20Notebook.pdf) <br>
utilize the chat format to have extended conversations with chatbots personalized or specialized for specific tasks or behaviors. <br>
prompt (mesages['user']) -> messages <br>
```python
messages =  [  
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},    
{'role':'user', 'content':'tell me a joke'},   
{'role':'assistant', 'content':'Why did the chicken cross the road'},   
{'role':'user', 'content':'I don\'t know'}  ]
```

You: messages['user'] <br>
Chatgpt: messages['assistant'] <br>
sets behaviour for assistant: messages['system'] <br>

[back](../gpt.md)