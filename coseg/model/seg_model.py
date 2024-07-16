import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class CLIPSegDecoder(nn.Module):
    def __init__(self, d_image=768, d_text=512, d_reduce=64, nhead=4, dropout=.1, dffn=20248):
        super().__init__()

        self.reduces = nn.ModuleList([
            nn.Linear(d_image, d_reduce) for _ in range(3)
        ])
        self.film_mul = nn.Linear(d_text, d_reduce)
        self.film_add = nn.Linear(d_text, d_reduce)

        self.decoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_reduce,
                    nhead=nhead,
                    dim_feedforward=dffn,
                    dropout=dropout,
                    activation=nn.GELU(),
                    batch_first=True,
                ),
                num_layers=1,
            )
            for _ in range(3)
        ])

        self.mask_head = nn.Sequential(
            nn.Conv2d(d_reduce, d_reduce, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce, d_reduce//2, kernel_size=(4, 4), stride=(4, 4)),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce//2, 1, kernel_size=(4, 4), stride=(4, 4))
        )

    def forward(self, lang_output, hidden_states):
        # Image sequence size
        self.image_seq_size = int(np.sqrt(hidden_states[0].shape[1]))
        
        masks = []
        for i, batch_embeddings in enumerate(lang_output.permute(1, 0, 2)):
            a  = None
            for hs, block, reduce in zip(hidden_states, self.decoder, self.reduces):
                hs = hs.permute(1, 0, 2)
                #hs = hs[:,1:,:].permute(1, 0, 2)
                if a is None:
                    a = reduce(hs)
                else:
                    a = a + reduce(hs)

                a = a * self.film_mul(batch_embeddings) + self.film_add(batch_embeddings)
                a = block(a)

            a = a[1:].permute(1, 2, 0)
            a = a.view(a.shape[0], a.shape[1], self.image_seq_size, self.image_seq_size)
            a = self.mask_head(a)
            masks.append(a)

        masks = torch.cat(masks, dim=1)
        return masks, lang_output