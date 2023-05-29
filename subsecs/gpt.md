---
layout: default
---

[back](../index.md)

# Generative pre-trained transformers (GPT)

deeplearning.ai course: [ChatGPT Prompt Engineering for Developers]()

[Improving language understanding with unsupervised learning](https://openai.com/research/language-unsupervised) <br>
Authors: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever<br>
Year: 2018

Goal of GPT: to learn universal representation of language that can be adapted to downstream tasks by task-specific fine-tuning.

Framework: **two-stage training procedure**
- unsupervised pre-training: on a large corpus of unlabeled text data
  - objective: predict the next word in a sequence of words given the preceding context.
  - outcome: learned parameters of NN & contextual word embedding.
- supervised fine-tuning: task-specific, adapt the model to a discriminative task with labeled data.

Model architecture: **transformer**
- Difference between GPT and transformer:
Transformer is a type of NN architecture that can be used for various tasks, 
while GPT is a specific implementation of this architecture. <br>
- Architectural difference: 

## Reference
[OpenAI GPT: Generative Pre-Training for Language Understanding](https://medium.com/dataseries/openai-gpt-generative-pre-training-for-language-understanding-bbbdb42b7ff4) <br>


[back](../index.md)