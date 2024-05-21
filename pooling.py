import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import json

class learnable_gaussian(nn.Module):
    '''
    I can weight something by a gaussian and let the network learn how wide the gaussian should be
    '''
    def __init__(self, center, var = 128, amp = 0.01):
        super().__init__()
        self.amp = nn.Parameter(data = torch.Tensor([amp]))
        self.var = nn.Parameter(data = torch.Tensor([var]))
        self.center = center

        self.device = torch.device('cpu')
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        self.to(self.device)

    def forward(self, x):
        return self.amp * torch.exp(-(x - self.center)**2/(2 * self.var))

class SwiGLU(nn.Module):
    '''
    Basically SwiGLU from ChatGPT, looks correct
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, use_layernorm = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(hidden_dim)
        
        self.device = torch.device('cpu')
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        self.to(self.device)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        output = self.output_layer(F.silu(x1) * x2)
        if self.use_layernorm:
            return self.ln(output)
        return output

class SwiGLUNetwork(nn.Module):
    '''
    Another piece of GPT code with minor modifications
    Since we just want a one sentence embedding I am just setting output_dim = 1
    '''
    def __init__(self, input_dim, mid_dim = 64, use_layernorm = True):
        super(SwiGLUNetwork, self).__init__()
        self.layers = nn.ModuleList()
        depth_dim = np.arange(start = 0, stop = input_dim+1, step = 128)
        depth_dim[0], depth_dim[-1] = mid_dim, input_dim
        depth_dim = depth_dim[::-1]

        # Add subsequent layers
        for i in range(len(depth_dim) - 1):
            self.layers.append(SwiGLU(depth_dim[i], depth_dim[i+1], depth_dim[i+1]))
        self.layers.append(SwiGLU(mid_dim, mid_dim, 1, use_layernorm = False))

        self.device = torch.device('cpu')
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class my_pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_512 = SwiGLUNetwork(512)
        self.ffn_384 = SwiGLUNetwork(384)
        self.ffn_256 = SwiGLUNetwork(256)
        self.ffn_128 = SwiGLUNetwork(128)
        
        self.gaussian_512 = learnable_gaussian(512)
        self.gaussian_384 = learnable_gaussian(384)
        self.gaussian_256 = learnable_gaussian(256)
        self.gaussian_128 = learnable_gaussian(128)

        #self.unconventional_pooling_weight = nn.Parameter(data =  torch.Tensor([0.01]))

        self.device = torch.device('cpu')
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        self.to(self.device)
    
    @staticmethod
    def linear_interpolate_sentence(masked_embeddings_tranposed, embed_length):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        embedding_512 = torch.empty(0).to(device)
        embedding_384 = torch.empty(0).to(device)
        embedding_256 = torch.empty(0).to(device)
        embedding_128 = torch.empty(0).to(device)
        
        for sentence, length in zip(masked_embeddings_tranposed, embed_length):
            sentence_512 = F.interpolate(sentence[:, 0:length].unsqueeze(0), size = 512, mode = 'linear')
            sentence_384 = F.interpolate(sentence[:, 0:length].unsqueeze(0), size = 384, mode = 'linear')
            sentence_256 = F.interpolate(sentence[:, 0:length].unsqueeze(0), size = 256, mode = 'linear')
            sentence_128 = F.interpolate(sentence[:, 0:length].unsqueeze(0), size = 128, mode = 'linear')

            embedding_512 = torch.cat((embedding_512, sentence_512), dim = 0)
            embedding_384 = torch.cat((embedding_384, sentence_384), dim = 0)
            embedding_256 = torch.cat((embedding_256, sentence_256), dim = 0)
            embedding_128 = torch.cat((embedding_128, sentence_128), dim = 0)
        
        return embedding_512, embedding_384, embedding_256, embedding_128


    def forward(self, features):
        # every sentence get's a mask that masks the rest of the length
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]
        embed_length = attention_mask.sum(1) # number from 0 to 512 which is embed length
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
        masked_embeddings = token_embeddings * input_mask_expanded # this is the real sentence
        mean_embeddings = torch.sum(masked_embeddings, 1)/torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        fnn_raw_embeddings = torch.transpose(token_embeddings, -2, -1) # put 512 in last dim for linear layer
        fnn_512_embeddings, fnn_384_embeddings, fnn_256_embeddings, fnn_128_embeddings = self.linear_interpolate_sentence(fnn_raw_embeddings, embed_length)
        
        sum_unconv_embed = (
            self.ffn_512(fnn_512_embeddings) * self.gaussian_512(embed_length)[:, None, None] +
            self.ffn_384(fnn_384_embeddings) * self.gaussian_384(embed_length)[:, None, None] +
            self.ffn_256(fnn_256_embeddings) * self.gaussian_256(embed_length)[:, None, None] + 
            self.ffn_128(fnn_128_embeddings) * self.gaussian_128(embed_length)[:, None, None])

        #output_mod = (sum_unconv_embed * self.unconventional_pooling_weight).squeeze()
        output_mod = (sum_unconv_embed).squeeze() 
        output = F.normalize(mean_embeddings + output_mod, p=2, dim=1)

        features.update({"sentence_embedding": output})
        return features


    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


    @staticmethod
    def load(input_path: str):
        weights = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        model = my_pooling()
        model.load_state_dict(weights)
        return model