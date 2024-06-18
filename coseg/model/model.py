from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch.nn.functional as F
from torch import nn
import torch
import warnings

from transformers import CLIPProcessor
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

    @configurable
    def __init__(
        self,
        d_reduce,
        nhead,
        nencoder,
        ndecoder,
        lang_model,
        alpha,
        beta,
        temp,
    ):
        
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

        super().__init__()
        self.predictor = CoSeg(
            d_reduce=d_reduce, 
            nhead=nhead, 
            nencoder=nencoder, 
            ndecoder=ndecoder, 
            lang_model=lang_model
        )
        warnings.warn("Model initialized...but no registered vocabulary! ")

    @classmethod
    def from_config(self, cfg):
        return {
            "d_reduce" : cfg.MODEL.D_REDUCE,
            "nhead" : cfg.MODEL.N_HEAD,
            "nencoder" : cfg.MODEL.N_ENCODER,
            "ndecoder" : cfg.MODEL.N_DECODER,
            "lang_model" : cfg.MODEL.LANG_MODEL,
            "alpha" : cfg.MODEL.ALPHA,
            "beta" : cfg.MODEL.BETA,
            "temp" : cfg.MODEL.TEMPERATURE
        }

    @property
    def device(self):
        return self.predictor.decoder.reduces[0].weight.device
    
    def criterion(self, mask_logits, label_logits, masks, ids):
        
        def dice_loss(y_true, y_pred):
            numerator = 2 * torch.sum(y_true * y_pred)
            denominator = torch.sum(y_true + y_pred)
            return 1 - numerator / denominator

        mask_loss_1 = nn.BCEWithLogitsLoss()
        mask_loss_2 = dice_loss
        lang_loss = nn.CrossEntropyLoss()

        l1 = mask_loss_1(mask_logits, masks)
        l2 = self.beta * mask_loss_2(masks, F.sigmoid(mask_logits))
        l3 = self.alpha * lang_loss(label_logits.permute(0, 2, 1), ids)

        return l1, l2, l3
    
    def register_vocabulary(self, label_embeddings):
        warnings.warn("Overwriting registered vocabulary")
        self.label_embeddings = label_embeddings.to(self.device)
    
    def inference(self, pixel_values):
        # Extract Pred
        mask_logits, pred_embeddings = self.predictor(pixel_values)
        # Prepare Pred
        predicted_embeddings = F.normalize(pred_embeddings, dim=-1).squeeze(0)
        all_class_probabilities = torch.softmax(predicted_embeddings @ self.label_embeddings.T / self.temp, dim=-1)
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

        return {"sem_seg": final_mask}

    def forward(self, batched_inputs):
        pixel_values = torch.stack([b['image'] for b in batched_inputs]).to(self.device)
        ids = torch.stack([b['ids'] for b in batched_inputs]).to(self.device)
        masks = torch.stack([b['masks'] for b in batched_inputs]).to(self.device)
        
        mask_logits, pred_embeddings = self.predictor(pixel_values)
        pred_embeddings = F.normalize(pred_embeddings, dim=-1)
        label_logits = pred_embeddings @ self.label_embeddings.T / self.temp

        l1, l2, l3 = self.criterion(mask_logits, label_logits, masks, ids)

        loss = {"loss": l1+l2+l3}
        return loss
