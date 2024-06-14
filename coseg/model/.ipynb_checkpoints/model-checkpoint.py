from detectron2.modeling import META_ARCH_REGISTRY
import torch.nn.functional as F
from torch import nn
import torch

from .seg_model import CLIPSegDecoder
from .lang_model import CLIPLang_xatten


class CoSeg(nn.Module):
    def __init__(self, d_reduce=64, nhead=4, nencoder=3, ndecoder=6, lang_model="prefix"):
        super().__init__()

        if lang_model == "prefix":
            self.encoders = CLIPLang_prefix(nhead=nhead, nencoder=nencoder+ndecoder)
        else:
            self.encoders = CLIPLang_xatten(nhead=nhead, nencoder=nencoder, ndecoder=ndecoder)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.decoder = CLIPSegDecoder(d_reduce=d_reduce, nhead=nhead)

    def update_vocabulary(self, vocabulary):
        # Get loss query table

        label_indices = list(vocabulary.keys())
        label_text = ["a photo of " + vocabulary[each] for each in label_indices]
        label_indices = self.processor(label_text, padding=True, return_tensors='pt').to(0)['input_ids']

        with torch.no_grad():
            label_embeddings = self.predictor.encoders.lang_model.text_model(label_indices)["pooler_output"]
            label_embeddings = self.predictor.encoders.lang_model.text_projector(label_embeddings)

        label_embeddings.requires_grad_(False)
        self.label_embeddings = label_embeddings
        self.label_indices = list(vocabulary.values())

    def forward(self, pixel_values):
        return self.decoder(*self.encoders(pixel_values, output_hidden_states=True))

@META_ARCH_REGISTRY.register()
class CoSeg_wrapper(nn.Module):
    '''
    A wrapper for CoSeg to be used for evaluation
    '''
    def __init__(self, cfg):

        d_reduce = cfg.MODEL.D_REDUCE
        nhead = cfg.MODEL.N_HEAD
        nencoder = cfg.MODEL.N_ENCODER
        ndecoder = cfg.MODEL.N_DECODER
        lang_model = cfg.MODEL.LANG_MODEL

        super().__init__()
        self.predictor = CoSeg(
            d_reduce=d_reduce, 
            nhead=nhead, 
            nencoder=nencoder, 
            ndecoder=ndecoder, 
            lang_model=lang_model
        )

    def criterion(self):
        pass
    
    def register_vocabulary(self, label_embeddings):
        self.label_embeddings = label_embeddings
    
    def inference(self, pixel_values):
        # Extract Pred
        mask_logits, pred_embeddings = self.predictor(pixel_values)
        # Prepare Pred
        predicted_embeddings = F.normalize(pred_embeddings, dim=-1).squeeze(0)
        all_class_probabilities = torch.softmax(predicted_embeddings @ self.label_embeddings.T / 0.08, dim=-1)
        predicted_ids = all_class_probabilities.argmax(dim=-1).cpu().numpy()
        predicted_ids = np.array([self.label_indices[id] for id in predicted_ids])

        class_probabilities = all_class_probabilities.max(dim=-1).values
        mask_probabilities = F.sigmoid(mask_logits.squeeze(0))

        intermediate_mask = torch.argmax(mask_probabilities, dim=0)

        final_mask = torch.zeros(intermediate_mask.shape)
        for id in predicted_ids:
            candidate_indices = np.where(predicted_ids == id)[0]
            if len(candidate_indices) == 0:
                continue

            candidate_probabilities = class_probabilities[candidate_indices].cpu().numpy()
            final_idx = np.argmax(candidate_probabilities)
            final_mask[intermediate_mask == candidate_indices[final_idx]] = id

        return final_mask

    def forward(self, pixel_values):
        pass
