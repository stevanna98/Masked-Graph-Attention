import torch.nn as nn

from modules_masked import MMAB, MSAB
from modules import PMA, SAB

class MaskedAttentionGraphs(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, 
                 dim_hidden=128, num_heads=4, ln=False):
        super(MaskedAttentionGraphs, self).__init__()
        self.encoder = nn.Sequential(
            MSAB(dim_input, dim_hidden, num_heads, ln=ln),
            MSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    # def forward(self, X, M):
    #     return self.decoder(self.encoder(X, M))
    # def forward(self, X, M):
    #     encoded = self.encoder[0](X, M)
    #     encoded = self.encoder[1](encoded, M)
    #     decoded = self.decoder(encoded)  
    #     return decoded
    def forward(self, X, M):
        for layer in self.encoder:
            X = layer(X, M)

        for layer in self.decoder:
            X = layer(X)

        return X