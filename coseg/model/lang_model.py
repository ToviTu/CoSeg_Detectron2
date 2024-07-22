from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch.nn.functional as F
from torch import nn
import torch
import math

class CLIPLang(nn.Module):
    def __init__(self, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()
        self.clip_version = clip_version
        self.text_model = CLIPTextModelWithProjection.from_pretrained(clip_version)

    def forward(self, **args):
        # text_embedding -> text_embeddings
        return self.text_model(**args)#['pooler_output']

class CLIPLang_xatten(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=4, ndecoder=4, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()

        clip = CLIPVisionModelWithProjection.from_pretrained(clip_version)

        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

        # Freeze vision encoder
        self.vision_model.requires_grad_(False)
        self.vision_model.post_layernorm.requires_grad_(True)

        # internal dimensions
        self.d_text = self.visual_projection.out_features
        self.d_image = self.visual_projection.in_features

        # Projection layer to language space
        self.visual_projection = nn.Linear(self.d_image, self.d_text, bias=False)

        # Learnable embeddings
        self.query_embeddings = nn.Embedding(20, self.d_text)
        # Optional: initialize query embeddings from [CLS] embedding
        self.class_embedding = self.vision_model.embeddings.class_embedding
        self.query_embeddings.weight.data.normal_(mean=0, std=0.02)

        # Learnable position embeddings for src & tgt
        self.target_position_embeddings = nn.Embedding(50, self.d_text)
        self.source_position_embeddings = nn.Embedding(512, self.d_text)

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = self.d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            dim_feedforward = 4*self.d_text if 4*self.d_text > 2048 else 2048,
            activation = nn.GELU(),
            batch_first = True,
            dropout = 0,
            norm_first = True,
            bias = False
        )
    
    def visual_forward(self, pixel_values):
        encoder_output = self.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states
        last_hidden_state = encoder_output.last_hidden_state[:, 1:, :]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)
        lang_tokens = self.visual_projection(last_hidden_state)

        return lang_tokens, hidden_states

    def forward(self, pixel_values, output_hidden_states = False):
        # Get visual features
        lang_tokens, hidden_states = self.visual_forward(pixel_values)

        # Initialize query tokens
        index_tensor = torch.arange(20).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        query_tokens = self.query_embeddings(index_tensor)

        # Positional encoding
        query_tokens = query_tokens + self.target_position_embeddings(index_tensor)
        src_index_tensor = torch.arange(lang_tokens.shape[1]).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        lang_tokens = lang_tokens + self.source_position_embeddings(src_index_tensor)

        # Decode texts
        text_pred = self.decoder(
                    lang_tokens,
                    query_tokens,
                )

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred


class CLIPLang_xatten_legacy(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=4, ndecoder=4, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()

        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_version)
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_version)

        # vision model
        self.vision_encoder = vision_model.vision_model
        self.vision_projector = vision_model.visual_projection

        # text model
        self.text_model = text_model.text_model
        self.text_projector = text_model.text_projection

        # internal dimensions
        self.d_text = self.text_model.embeddings.token_embedding.weight.shape[1]
        self.d_image = self.vision_encoder.embeddings.position_embedding.weight.shape[1]

        # Learnable embeddings
        self.query_embeddings = nn.Embedding(20, self.d_text)
        self.query_embeddings.weight.data.normal_(mean=0, std=0.02)

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = self.d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            dim_feedforward = 2048,
            activation = nn.GELU(),
            batch_first = True
        )

    def to_embedding(self, input_ids):
        # index -> text_embeddings
        return self.text_model.embeddings.token_embedding(input_ids)

    def position_encode(self, embeddings):
        index_tensor = torch.arange(embeddings.shape[1]).repeat(embeddings.shape[0], 1).to(embeddings.device)
        return embeddings + self.text_model.embeddings.position_embedding(index_tensor)

    def visual_forward(self, pixel_values, output_hidden_states=False):
        # image -> text_embeddings
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        # text_embedding -> text_embeddings
        return self.text_model.encoder(text_embeddings)

    def decoder_forward(self, img_src, txt_tgt):
        # Send image seq and text seq to lang model
        return self.decoder(
            img_src,
            txt_tgt,
        )

    def cond_forward(self, pixel_values, output_hidden_states = False):
        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Remove the [cls] token
        visual_features = visual_features[:, 1:, :]

        # Initialize query tokens
        index_tensor = torch.arange(20).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        query_tokens = self.query_embeddings(index_tensor)
        query_tokens = self.position_encode(query_tokens)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, query_tokens)

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, output_hidden_states=output_hidden_states)
            return text_pred, hidden_states

        return self.cond_forward(pixel_values)