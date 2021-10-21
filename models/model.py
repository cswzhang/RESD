# coding=utf-8
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .VAE import VAE


class RESD(nn.Module):
    # 对抗里去掉R的部分
    def __init__(self, config, graph, features):
        super(RESD, self).__init__()
        self.config = config
        self.G = graph

        self.features = torch.from_numpy(features).to(self.config.device, dtype=torch.float32)

        self.config.struct[0] = self.features.shape[1]
        self.degree = torch.from_numpy(np.array(sorted(dict(self.G.degree).items(), key=lambda x: x[0]))[:, 1]).to(self.config.device,
                                                                   dtype=torch.float32).reshape(
            -1, 1)
        self.degree = torch.log(self.degree + 1)
        self.vae = VAE(self.G, self.config).to(self.config.device, dtype=torch.float32)
        self.mlp = nn.ModuleList([
            nn.Linear(self.config.struct[-1], self.config.struct[-1]),
            nn.Linear(self.config.struct[-1], 1)
        ]).to(self.config.device, dtype=torch.float32)
        for i in range(len(self.mlp)):
            nn.init.xavier_uniform_(self.mlp[i].weight)
            nn.init.uniform_(self.mlp[i].bias)
        self.mseLoss = nn.MSELoss()
        self.bceLoss = nn.BCEWithLogitsLoss()

    def generate_fake(self, h_state):
        z = torch.from_numpy(np.random.normal(0, 1, size=h_state.size())).to(
            self.config.device, dtype=torch.float32)
        return z

    def mlp_out(self, embedding):
        for i, layer in enumerate(self.mlp):
            embedding = torch.relu(layer(embedding))
        return embedding

    def forward(self, input_):
        features = self.features[input_]
        mu, sigma, embedding, vae_out = self.vae(features)
        vae_loss = self.config.alpha * F.mse_loss(vae_out, features)
        guide_loss = self.config.gamma * F.l1_loss(self.mlp_out(embedding), self.degree[input_])
        return vae_loss + guide_loss

    def get_embedding(self):
        return self.vae.get_embedding(self.features)
