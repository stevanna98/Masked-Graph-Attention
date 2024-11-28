import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def masked_attention(self, A, M):
        if M is not None:
            M = M.repeat(self.num_heads, 1, 1)
            A = A.masked_fill(M == 0, float('-inf'))
        return A
    
    def forward(self, Q, K, M):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Masked attention
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = self.masked_attention(A, M)
        A = torch.softmax(A, 2) 

        # Output
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    
class MSAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(MSAB, self).__init__()
        self.mab = MMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, M):
        return self.mab(X, X, M)
        

