import torch.nn as nn

from modules_masked import MMAB, MSAB
from modules import PMA, SAB

class MaskedAttentionGraphs(nn.Module):
    def __init__(self, dim_input, dim_output, 
                 num_seeds=1, dim_hidden=64, num_heads=4, ln=False):
        super(MaskedAttentionGraphs, self).__init__()
        self.encoder = nn.Sequential(
            MSAB(dim_input, dim_hidden, num_heads, ln=ln),
            MSAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            MSAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.sab = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, num_seeds, ln=ln),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid()
        )

    def forward(self, X, M):
        encoded = self.encoder[0](X, M)
        for layer in self.encoder[1:]:
            encoded = layer(encoded, M)
        encoded = self.sab(encoded)
        return self.decoder(encoded)