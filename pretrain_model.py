
from typing import List, Tuple, Optional, Union
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    SiglipVisionModel,
    BertForMaskedLM,
    BatchEncoding,
    SiglipModel
)
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPooler

from token_compression import TokenCompressionAdapter
from utils import get_world_size, all_gather_object_safe, get_rank, concat_all_gather

VISION_TOKEN_DICT = {
    "google/siglip2-base-patch16-224": 196,
    "google/siglip2-base-patch16-384": 576,
    "google/siglip2-large-patch16-384": 576,
    "google/siglip2-large-patch16-256": 256
}
IMAGE_TOKEN = "<image>"
MAX_TEXT_CONTEXT_LENGTH = 64
MAX_SIGLIP_TEXT_CONTEXT_LENGTH = 64

class MultimodalPretrainModel(nn.Module):
    def __init__(
        self,                 
        siglip_path: str,
        base_language_model_path: str,
        multimodal_projection_hidden_dim: int = 8192,
        num_negatives_per_sample: int = 1,
        finetune: bool = False,
        num_compressed_tokens: Optional[int] = None
    ):        
        super().__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained(siglip_path)
        self.finetune = finetune
        self.tokenizer = AutoTokenizer.from_pretrained(base_language_model_path, extra_special_tokens={"image_token": IMAGE_TOKEN})
        self.language_model = BertForMaskedLM.from_pretrained(base_language_model_path)
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.text_embeddings = self.language_model.get_input_embeddings()

        use_token_compression = (num_compressed_tokens is not None)
        self.max_vision_context_length = num_compressed_tokens if use_token_compression else VISION_TOKEN_DICT[siglip_path]
        self.max_context_length = MAX_TEXT_CONTEXT_LENGTH + self.max_vision_context_length
        self._maybe_extend_position_embeddings(target_max_positions=self.max_context_length)

        if use_token_compression:
            self.vision_projection = TokenCompressionAdapter(num_compressed_tokens=num_compressed_tokens,
                                                             hidden_size=self.vision_encoder.config.hidden_size,
                                                             intermediate_size=multimodal_projection_hidden_dim,
                                                             output_size=self.language_model.config.hidden_size,
                                                             hidden_act=self.language_model.config.hidden_act,
                                                             num_attention_heads=self.vision_encoder.config.num_attention_heads,
                                                             layer_norm_eps=self.vision_encoder.config.layer_norm_eps)
        else:
            self.vision_projection = nn.Sequential(
                nn.Linear(self.vision_encoder.config.hidden_size, multimodal_projection_hidden_dim),
                ACT2FN[self.language_model.config.hidden_act],
                nn.Linear(multimodal_projection_hidden_dim, self.language_model.config.hidden_size)
            )

        self.clip = SiglipModel.from_pretrained(siglip_path).eval()
        self.siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_path)

        self.classification_head = nn.Sequential(
            BertPooler(self.language_model.config),
            nn.Linear(self.language_model.config.hidden_size, 1)
        )

        self.text_projection = nn.Linear(
            self.language_model.config.hidden_size,
            self.clip.config.text_config.projection_size
        )

        self.num_negatives_per_sample = num_negatives_per_sample
        self.loss_fct = nn.CrossEntropyLoss()

        
    def forward(
            self,
            pixel_values: torch.Tensor,
            captions: List[str],
            img_ids: Optional[List[int]] = None,
            teacher_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:

        # 1. Compute visual features
        with torch.inference_mode():
            vision_encoder_output_tokens = self.vision_encoder(pixel_values=pixel_values).last_hidden_state

        vision_projected_tokens = self.vision_projection(vision_encoder_output_tokens.clone())

        with torch.inference_mode():
            teacher_vision_encoder_output_tokens = teacher_model.vision_encoder(pixel_values=pixel_values).last_hidden_state
            teacher_vision_projected_tokens = teacher_model.vision_projection(teacher_vision_encoder_output_tokens)

        # assert vision_projected_tokens.shape[1] == self.max_vision_context_length

        # 2. Masked Language modeling loss
        if not self.finetune:
            lm_loss = self.compute_lm_loss(captions, vision_projected_tokens)
            itc_loss = self.compute_itc_loss(captions, device=vision_projected_tokens.device)
        else:
            lm_loss = torch.zeros(1, device=pixel_values.device, requires_grad=True)
            itc_loss = torch.zeros(1, device=pixel_values.device, requires_grad=True)

        # 4. Image-Text Matching loss
        itm_loss, distil_loss = self.compute_itm_loss(pixel_values, captions, vision_projected_tokens, img_ids, teacher_model=teacher_model, teacher_vision_projected_tokens=teacher_vision_projected_tokens)

        return lm_loss, itc_loss, itm_loss, distil_loss

    def compute_distil_loss(self, captions, vision_projected_tokens, teacher_model, pixel_values):
        pos_logits = self.compute_logits(captions=captions, vision_projected_tokens=vision_projected_tokens)
        with torch.inference_mode():
            teacher_logits = teacher_model.compute_logits(captions=captions, pixel_values=pixel_values)
        loss_fn = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()
        loss_distil = loss_fn(pos_logits, sigmoid(teacher_logits).detach())
        return loss_distil

    def compute_lm_loss(self, captions: List[str], vision_embeds: torch.FloatTensor) -> torch.FloatTensor:
        text_inputs, labels = self.prepare_language_modeling_inputs_labels(captions, vision_embeds)
        outputs = self.language_model(**text_inputs)
        logits = outputs.logits
        lm_loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return lm_loss

    def compute_itc_loss(self, captions: List[str], device: torch.device):
        clip_text_inputs = self.siglip_tokenizer(
            captions,
            padding="max_length",
            max_length=MAX_TEXT_CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            clip_text_embeds = self.clip.get_text_features(**clip_text_inputs)

        text_inputs = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_CONTEXT_LENGTH,
            return_tensors="pt"
        ).to(device)

        outputs = self.language_model(**text_inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        text_embeds = self.text_projection(last_hidden_state[:, 0, :])

        itc_loss = (
                1 - torch.cosine_similarity(text_embeds, clip_text_embeds.clone())
        ).mean()
        return itc_loss

    def compute_itm_loss(
            self,
            pixel_values: torch.FloatTensor,
            captions: List[str],
            vision_projected_tokens: torch.FloatTensor,
            img_ids: Optional[List[int]] = None,
            teacher_model: Optional[torch.nn.Module] = None,
            teacher_vision_projected_tokens: Optional[torch.FloatTensor] = None,
    ):
        distil_loss_fn = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()

        pos_logits = self.compute_logits(captions, vision_projected_tokens=vision_projected_tokens).view(-1, 1)
        bs = pos_logits.shape[0]

        all_captions = [None] * get_world_size()
        all_gather_object_safe(all_captions, captions)
        all_captions = sum(all_captions, [])

        if img_ids is not None:
            all_img_ids = [None] * get_world_size()
            all_gather_object_safe(all_img_ids, img_ids)
            all_img_ids = sum(all_img_ids, [])
        else:
            all_img_ids = None


        all_vision_projected_tokens = concat_all_gather(vision_projected_tokens)
        all_teacher_vision_projected_tokens = concat_all_gather(teacher_vision_projected_tokens)

        with torch.inference_mode():
            teacher_pos_logits = teacher_model.compute_logits(captions=captions, vision_projected_tokens=teacher_vision_projected_tokens)
        loss_distil_pos = distil_loss_fn(pos_logits.view(-1), sigmoid(teacher_pos_logits).detach())

        neg_img_idxs, neg_text_idxs = self.sample_in_batch_negatives(
            captions,
            pixel_values,
            all_captions,
            all_img_ids,
            self.num_negatives_per_sample
        )

        negative_vision_projected_tokens = all_vision_projected_tokens[neg_img_idxs.clone().view(-1)]
        teacher_negative_vision_projected_tokens = all_teacher_vision_projected_tokens[neg_img_idxs.clone().view(-1)]
        repeated_captions = list(np.repeat(captions, self.num_negatives_per_sample))
        neg_logits_images = self.compute_logits(
            repeated_captions,
            vision_projected_tokens=negative_vision_projected_tokens
        ).view(bs, -1)
        itm_logits_images = torch.cat((pos_logits, neg_logits_images), dim=1)
        itm_loss_images = self.loss_fct(itm_logits_images, torch.zeros(bs, dtype=torch.long, device=pos_logits.device))

        with torch.inference_mode():
            teacher_neg_logits_images = teacher_model.compute_logits(captions=repeated_captions, vision_projected_tokens=teacher_negative_vision_projected_tokens)
        loss_distil_neg_images = distil_loss_fn(neg_logits_images.view(-1), sigmoid(teacher_neg_logits_images.view(-1)).detach())

        negative_captions = [all_captions[neg_idx] for neg_idx in neg_text_idxs.clone().view(-1)]
        repeated_vision_projected_tokens = vision_projected_tokens.repeat_interleave(self.num_negatives_per_sample, dim=0)
        neg_logits_texts = self.compute_logits(
            negative_captions,
            vision_projected_tokens=repeated_vision_projected_tokens
        ).view(bs, -1)
        itm_logits_texts = torch.cat((pos_logits, neg_logits_texts), dim=1)
        itm_loss_texts = self.loss_fct(itm_logits_texts, torch.zeros(bs, dtype=torch.long, device=pos_logits.device))

        with torch.inference_mode():
            teacher_repeated_vision_projected_tokens = teacher_vision_projected_tokens.repeat_interleave(self.num_negatives_per_sample,
                                                                                         dim=0)
            teacher_neg_logits_texts = teacher_model.compute_logits(captions=negative_captions, vision_projected_tokens=teacher_repeated_vision_projected_tokens)
        loss_distil_neg_texts = distil_loss_fn(neg_logits_texts.view(-1), sigmoid(teacher_neg_logits_texts.view(-1)).detach())

        itm_loss = (itm_loss_images + itm_loss_texts) / 2
        distil_loss = (loss_distil_pos + loss_distil_neg_images + loss_distil_neg_texts) / 3
        return itm_loss, distil_loss


    def compute_logits(self,
            captions: List[str],
            pixel_values: Optional[torch.FloatTensor] = None,
            vision_projected_tokens: Optional[torch.FloatTensor] = None,
            vision_encoder_output_tokens: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        if vision_projected_tokens is None:
            if vision_encoder_output_tokens is None:
                with torch.inference_mode():
                    vision_encoder_output_tokens_ = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
                vision_encoder_output_tokens = vision_encoder_output_tokens_.clone()
            vision_projected_tokens = self.vision_projection(vision_encoder_output_tokens)

        inputs = self.prepare_language_model_inputs(captions, vision_projected_tokens)
        hidden_states = self.language_model(**inputs, output_hidden_states=True).hidden_states[-1]
        logits = self.classification_head(hidden_states).view(-1)
        return logits


    def prepare_language_modeling_inputs_labels(
            self,
            captions: List[str],
            vision_embeds: torch.FloatTensor
    ) -> Tuple[dict, torch.LongTensor]:
        
        target_text_inputs = self.tokenize_text_image_pair(captions, vision_embeds)
        target_text_inputs = dict(target_text_inputs)
        labels = target_text_inputs["input_ids"]

        text_inputs = copy.deepcopy(target_text_inputs)
        input_ids = text_inputs["input_ids"]

        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
            for seq in input_ids
        ], dtype=torch.bool, device=input_ids.device)
        random_mask = (torch.rand(input_ids.shape) < 0.5).to(input_ids.device)
        final_mask = (input_ids != self.tokenizer.image_token_id) & ~special_tokens_mask & random_mask

        masked_input_ids = input_ids.clone()
        masked_input_ids[final_mask] = self.tokenizer.mask_token_id
        text_inputs["input_ids"] = masked_input_ids
        labels[~final_mask] = -100 # Set labels to -100 where we do NOT want to compute loss

        inputs_embeds = self.compute_inputs_embeds(text_inputs["input_ids"], vision_embeds)
        text_inputs["inputs_embeds"] = inputs_embeds
        text_inputs.pop("input_ids")

        return text_inputs, labels


    def tokenize_text_image_pair(self, captions: List[str], vision_embeds: torch.FloatTensor) -> BatchEncoding:
        bs, vision_seq_len, _ = vision_embeds.shape
        image_placeholder = "".join([IMAGE_TOKEN] * vision_seq_len)
        text_inputs = self.tokenizer(
            captions,
            [image_placeholder] * bs,
            padding=True,
            truncation="only_first",
            max_length=self.max_context_length,  # Limit total sequence length to BERT's maximum
            return_tensors="pt"
        ).to(vision_embeds.device)
        return text_inputs

    def compute_inputs_embeds(self, input_ids: torch.LongTensor, vision_embeds: torch.FloatTensor) -> torch.FloatTensor:
        inputs_embeds = self.text_embeddings(input_ids)
        mask = (input_ids == self.tokenizer.image_token_id)
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, vision_embeds)
        return inputs_embeds

    def prepare_language_model_inputs(self, captions: List[str], vision_embeds: torch.FloatTensor) -> dict:
        tokenized_pair = self.tokenize_text_image_pair(captions, vision_embeds)
        tokenized_pair = dict(tokenized_pair)
        inputs_embeds = self.compute_inputs_embeds(
            input_ids=tokenized_pair.pop("input_ids"),
            vision_embeds=vision_embeds
        )
        tokenized_pair["inputs_embeds"] = inputs_embeds
        return tokenized_pair

    def _maybe_extend_position_embeddings(self, target_max_positions: int) -> None:
        """Extend BERT-like learned position embeddings to at least target_max_positions.

        Copies existing weights and repeats the last position for the new range.
        """
        base_model = self.language_model.bert
        embeddings = base_model.embeddings

        pos_emb: nn.Embedding = embeddings.position_embeddings
        old_max_positions, hidden_size = pos_emb.weight.shape

        if old_max_positions >= target_max_positions:
            return

        device = pos_emb.weight.device
        dtype = pos_emb.weight.dtype

        new_pos_emb = nn.Embedding(target_max_positions, hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            new_pos_emb.weight[:old_max_positions] = pos_emb.weight
            new_pos_emb.weight[old_max_positions:] = pos_emb.weight[-1:].repeat(target_max_positions - old_max_positions, 1)

        embeddings.position_embeddings = new_pos_emb

        # Update config and position_ids buffer if present
        self.language_model.config.max_position_embeddings = target_max_positions

        embeddings.register_buffer(
            "position_ids",
            torch.arange(target_max_positions).expand((1, -1)),
            persistent=False,
        )

        self.tokenizer.model_max_length = max(int(getattr(self.tokenizer, "model_max_length", 0) or 0), target_max_positions)

    @torch.inference_mode()
    def sample_in_batch_negatives(
            self,
            captions: List[str],
            pixel_values: torch.Tensor,
            all_captions: List[str],
            all_img_ids: Optional[List[str]] = None,
            num_negatives_per_sample: int = 1
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:

        if all_img_ids is None:
            all_img_ids = list(range(len(all_captions)))

        clip_text_inputs = self.siglip_tokenizer(
            captions,
            padding="max_length",
            max_length=MAX_SIGLIP_TEXT_CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(pixel_values.device)

        outputs = self.clip(pixel_values=pixel_values, **clip_text_inputs)

        text_embeds, image_embeds = outputs.text_embeds, outputs.image_embeds
        all_text_embeds = concat_all_gather(text_embeds)
        all_image_embeds = concat_all_gather(image_embeds)

        # cosine similarity as logits
        logits_per_text = text_embeds @ all_image_embeds.T
        logits_per_image = image_embeds @ all_text_embeds.T

        bs = text_embeds.shape[0]
        rank = get_rank()

        equal_images_mask = self.get_equal_objects_mask(all_img_ids)[bs * rank: bs * (rank + 1)].to(pixel_values.device)
        negative_image_sampling_weights = logits_per_text.masked_fill(equal_images_mask, torch.finfo(logits_per_text.dtype).min)
        negative_image_sampling_weights = F.softmax(negative_image_sampling_weights, dim=-1)

        global_negative_image_idxs = torch.topk(
            negative_image_sampling_weights,
            k=num_negatives_per_sample,
            dim=-1
        ).indices
        equal_captions_mask = self.get_equal_objects_mask(all_captions)[bs * rank: bs * (rank + 1)].to(pixel_values.device)
        negative_text_sampling_weights = logits_per_image.masked_fill(equal_captions_mask, torch.finfo(logits_per_image.dtype).min)
        negative_text_sampling_weights = F.softmax(negative_text_sampling_weights, dim=-1)


        global_negative_text_idxs = torch.topk(
            negative_text_sampling_weights,
            k=num_negatives_per_sample,
            dim=-1
        ).indices

        return global_negative_image_idxs, global_negative_text_idxs

    @staticmethod
    def get_equal_objects_mask(objects: list) -> torch.BoolTensor:
        object_to_id = {obj: idx for idx, obj in enumerate(set(objects))}
        objects_ids = torch.tensor([object_to_id[obj] for obj in objects])
        mask = (objects_ids[:, None] == objects_ids[None, :])
        return mask
