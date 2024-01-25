import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Skipgram(nn.Module):
    
    def __init__(self, voc_size, emb_size, word2index):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)

        self.word2index = word2index
    
    def forward(self, center, outside, all_vocabs):
        center_embedding     = self.embedding_center(center)  #(batch_size, 1, emb_size)
        outside_embedding    = self.embedding_center(outside) #(batch_size, 1, emb_size)
        all_vocabs_embedding = self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size)
        
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1) 

        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)  #(batch_size, 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))  #scalar
        
        return loss
    
    def get_embed(self, word):
        word2index = self.word2index
        
        try:
            index = word2index[word]
        except:
            index = word2index['<UNK>']
            
        word = torch.LongTensor([index])
        
        embed_c = self.embedding_center(word)
        embed_o = self.embedding_outside(word)
        embed   = (embed_c + embed_o) / 2
        
        return embed[0][0].item(), embed[0][1].item()

class SkipgramNeg(nn.Module):
    
    def __init__(self, voc_size, emb_size, word2index):
        super(SkipgramNeg, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid        = nn.LogSigmoid()

        self.word2index = word2index
    
    def forward(self, center, outside, negative):
        #center, outside:  (bs, 1)
        #negative       :  (bs, k)
        
        center_embed   = self.embedding_center(center) #(bs, 1, emb_size)
        outside_embed  = self.embedding_outside(outside) #(bs, 1, emb_size)
        negative_embed = self.embedding_outside(negative) #(bs, k, emb_size)
        
        uovc           = outside_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, 1)
        ukvc           = -negative_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, k)
        ukvc_sum       = torch.sum(ukvc, 1).reshape(-1, 1) #(bs, 1)
        
        loss           = self.logsigmoid(uovc) + self.logsigmoid(ukvc_sum)
        
        return -torch.mean(loss)
    
    def get_embed(self, word):
        word2index = self.word2index
        
        try:
            index = word2index[word]
        except:
            index = word2index['<UNK>']
            
        word = torch.LongTensor([index])
        
        embed_c = self.embedding_center(word)
        embed_o = self.embedding_outside(word)
        embed   = (embed_c + embed_o) / 2
        
        return embed[0][0].item(), embed[0][1].item()
    

class Glove(nn.Module):
    
    def __init__(self, voc_size, emb_size, word2index):
        super(Glove, self).__init__()
        self.center_embedding  = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        
        self.center_bias       = nn.Embedding(voc_size, 1) 
        self.outside_bias      = nn.Embedding(voc_size, 1)

        self.word2index = word2index


    def forward(self, center, outside, coocs, weighting):
        center_embeds  = self.center_embedding(center) #(batch_size, 1, emb_size)
        outside_embeds = self.outside_embedding(outside) #(batch_size, 1, emb_size)
        
        center_bias    = self.center_bias(center).squeeze(1)
        target_bias    = self.outside_bias(outside).squeeze(1)
        
        inner_product  = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #(batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1)
        
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
        
        return torch.sum(loss)
    
    #let's write a function to get embedding given a word
    def get_embed(self, word):
        word2index = self.word2index
        
        try:
            index = word2index[word]
        except:
            index = word2index['<UNK>']
            
        word = torch.LongTensor([index])
        
        embed_c = self.center_embedding(word)
        embed_o = self.outside_embedding(word)
        embed   = (embed_c + embed_o) / 2
        
        return embed[0][0].item(), embed[0][1].item()
    

class CustomGensim():
    def __init__(self, model):
        self.model = model

    def get_embed(self, word):

        default_vector = np.zeros(self.model.vector_size)
        
        try: 
            result = self.model.get_vector(word.lower().strip())
        except:
            result = default_vector

        return result