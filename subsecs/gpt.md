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

## Framework: **two-stage training procedure**
- unsupervised pre-training: on a large corpus of unlabeled text data
  - objective: predict the next word in a sequence of words given the preceding context.
  - outcome: learned parameters of NN & contextual word embedding.
- supervised fine-tuning: task-specific, adapt the model to a discriminative task with labeled data.

## Model architecture: **transformer**
- Difference between GPT and transformer:
Transformer is a type of NN architecture that can be used for various tasks, 
while GPT is a specific implementation of this architecture. <br>
- Architectural difference: 
  - a **transformer decoder**: unidirectional self-attentive model, uses only the tokens *preceding* a given token in the 
  sequence to attend that token (*auto-regressive (AR)*). ([BERT](../subsecs/bert.md) is transformer encoder: bidirectional self-attentive model, 
  uses *all* the tokens in a sequence to attend each token in that sequence.)

> GPT's transformer decoder architecture: 
> ![gpt](../pics/gpt.JPG) 

### Stage 1: unsupervised pre-training
Maximise objective function: <br>
![gpt_pretrain](../pics/gpt_pretrain.JPG) <br>
- U: a given corpus contains n tokens
- u_i: i-th token 
- k: window size, i.e., previous k tokens
- Θ: model parameters

Attention and probability are calculated through: <br>
![gpt_att](../pics/gpt_att.JPG) <br>
- W_e: embedding weights
- W_p positional encodings
- state h_l is calculated using state h_(l-1)
- TODO: add explanation to calculation

### Stage 2: supervised fine-tuning


## Reference
[OpenAI GPT: Generative Pre-Training for Language Understanding](https://medium.com/dataseries/openai-gpt-generative-pre-training-for-language-understanding-bbbdb42b7ff4) <br>


[back](../index.md)