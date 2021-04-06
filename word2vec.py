import torch.nn as nn
import torch.nn.functional as F
import torch

class Word2Vec(nn.Module):

    def __init__(self, vocabularySize, features):
        super(Word2Vec, self).__init__()

        #in: vocabularySize, out: features
        self.embeddings = nn.Linear(vocabularySize, features, bias=False)
        nn.init.normal_(self.embeddings.weight)
        #in: features, out: vocabularySize
        self.outputLayer = nn.Linear(features, vocabularySize, bias=False)
        nn.init.normal_(self.outputLayer.weight)

    def forward(self, inputVector):
        return self.outputLayer(self.embeddings(inputVector))

    def printWeights(self):
        print('Hidden layer shape:', self.embeddings.weight.shape, "Output layer shape:", self.outputLayer.weight.shape)
        print('Hidden layer:\n', self.embeddings.weight, '\nOutput layer', self.outputLayer.weight)

    def initLayers(self):
        nn.init.normal_(self.embeddings.weight)
        nn.init.normal_(self.outputLayer.weight)

    def similarity(self, firstIndex, secondIndex, vocabularySize, device):
        firstVector = torch.zeros(vocabularySize, device=device)
        firstVector[firstIndex] = 1
        secondVector = torch.zeros(vocabularySize, device=device)
        secondVector[secondIndex] = 1

        firstOutput = self.embeddings(firstVector)
        secondOutput = self.embeddings(secondVector)

        cos = nn.CosineSimilarity(dim=0)
        return cos(firstOutput, secondOutput)
