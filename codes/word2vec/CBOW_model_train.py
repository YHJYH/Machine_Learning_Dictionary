"""
source exercise: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#exercise-computing-word-embeddings-continuous-bag-of-words
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10 # a hyperparameter
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
data = []

for i in range(CONTEXT_SIZE, len(raw_text)-CONTEXT_SIZE):
    context = (
        [raw_text[i-j-1] for j in range(CONTEXT_SIZE)] + [raw_text[i+j+1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, [target]))

class CBOW(nn.Module):

    def __init__(self, context_size, embedding_size, vocab_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) # projection layer
        self.linear = nn.Linear(embedding_size * context_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs).view((1, -1))
        output = self.linear(embedded)
        return output

def make_context_vector(context):
    idxs = [word_to_idx[i] for i in context]
    return torch.tensor(idxs, dtype=torch.long)

losses = []
criterion = nn.CrossEntropyLoss()
model = CBOW(CONTEXT_SIZE*2, EMBEDDING_DIM, vocab_size)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    for context, target in data:
        context_idxs = make_context_vector(context)
        target_idx = make_context_vector(target)
        model.zero_grad()
        word_embedded = model(context_idxs)
        loss = criterion(word_embedded, target_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss
    losses.append(total_loss)
    print(f"current epoch: {epoch+1}, current loss: {total_loss}")

print(f"word embedding for word {'abstract'} is {model.embedding(make_context_vector(['abstract']))}")
