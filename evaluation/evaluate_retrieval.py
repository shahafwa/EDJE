import argparse
from typing import Tuple

from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipModel, AutoTokenizer, SiglipTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset
from data.flickr30k_dataset import flickr30k_retrieval_eval
from data.coco_karpathy_dataset import coco_karpathy_retrieval_eval
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML

from pretrain_model import MultimodalPretrainModel

K_LIST = [1, 5, 10]

@torch.inference_mode()
def clip_evaluation_t2i(test_dataset: Dataset, device: torch.device, config: dict):
    siglip_path = config['siglip_path']
    clip = SiglipModel.from_pretrained(siglip_path).eval().to(device)
    siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_path)

    image_embeds, image_token_embeds = compute_image_embeds(clip, test_dataset, device, config)
    caption_embeds = compute_caption_embeds(clip, siglip_tokenizer, test_dataset, device, config)
    top_k_indices = get_embeddings_top_k(caption_embeds, image_embeds, config['k'])

    clip_metrics = {f"R@{k}": 0 for k in K_LIST}

    for idx, text in enumerate(test_dataset.text):
        target_image_idx = test_dataset.txt2img[idx]

        clip_top_k_img_idxs = top_k_indices[idx]

        for k in K_LIST:
            clip_metrics[f"R@{k}"] += int(target_image_idx in clip_top_k_img_idxs[:k])
    num_samples = len(test_dataset.text)

    clip_metrics = {k: v / num_samples for k, v in clip_metrics.items()}
    print(f"CLIP Metrics: {clip_metrics}")
    return clip_metrics


@torch.inference_mode()
def evaluation_t2i(model: MultimodalPretrainModel, test_dataset: Dataset, device: torch.device, config: dict):
    siglip_path = config['siglip_path']
    clip = SiglipModel.from_pretrained(siglip_path).eval().to(device)
    siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_path)

    image_embeds, image_token_embeds = compute_image_embeds(clip, test_dataset, device, config)
    caption_embeds = compute_caption_embeds(clip, siglip_tokenizer, test_dataset, device, config)
    top_k_indices = get_embeddings_top_k(caption_embeds, image_embeds, config['k'])

    clip_metrics = {f"R@{k}": 0 for k in K_LIST}
    reranked_metrics = {f"R@{k}": 0 for k in K_LIST}

    pbar = tqdm(total=len(test_dataset.text), desc="Evaluating")

    for idx, text in enumerate(test_dataset.text):
        target_image_idx = test_dataset.txt2img[idx]

        clip_top_k_img_idxs = top_k_indices[idx]
        imgs_to_rerank_vision_features = image_token_embeds[clip_top_k_img_idxs]
        scores = model.compute_logits(
            captions=[text] * imgs_to_rerank_vision_features.shape[0],
            vision_encoder_output_tokens=imgs_to_rerank_vision_features.to(device)
        )
        reranked_top_k_img_idxs = [
            img_idx for
            img_idx, _ in
            sorted(zip(clip_top_k_img_idxs, scores), key=lambda x: x[1], reverse=True)
        ]

        for k in K_LIST:
            clip_metrics[f"R@{k}"] += int(target_image_idx in clip_top_k_img_idxs[:k])
            reranked_metrics[f"R@{k}"] += int(target_image_idx in reranked_top_k_img_idxs[:k])

        pbar.set_description(
            f"CLIP R@1: {clip_metrics['R@1'] / (idx + 1):.4f} | Reranked R@1: {reranked_metrics['R@1'] / (idx + 1):.4f}")
        pbar.update(1)

    num_samples = len(test_dataset.text)
    clip_metrics = {k: v / num_samples for k, v in clip_metrics.items()}
    reranked_metrics = {k: v / num_samples for k, v in reranked_metrics.items()}
    print(f"CLIP Metrics: {clip_metrics}")
    print(f"Reranked Metrics: {reranked_metrics}")
    return reranked_metrics


@torch.inference_mode()
def compute_image_embeds(clip: SiglipModel, test_dataset: Dataset, device: torch.device,
                         config: dict) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    img_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False
    )

    img_embeds, img_token_embeds = [], []
    for images_batch, _ in tqdm(img_loader, desc="Computing image embeddings", total=len(img_loader)):
        images_batch = images_batch.to(device, non_blocking=True)
        batch_img_outputs = clip.vision_model(images_batch)
        batch_embeds, batch_token_embeds = batch_img_outputs.pooler_output, batch_img_outputs.last_hidden_state
        batch_embeds = F.normalize(batch_embeds, dim=-1)
        img_embeds.append(batch_embeds.cpu())
        img_token_embeds.append(batch_token_embeds.cpu())
    img_embeds = torch.cat(img_embeds, dim=0)
    img_token_embeds = torch.cat(img_token_embeds, dim=0)
    return img_embeds, img_token_embeds


@torch.inference_mode()
def compute_caption_embeds(clip: SiglipModel, siglip_tokenizer: PreTrainedTokenizer, test_dataset: Dataset,
                           device: torch.device, config: dict) -> torch.FloatTensor:
    texts = test_dataset.text
    texts_batches = [texts[i:i + config['batch_size']] for i in range(0, len(texts), config['batch_size'])]

    caption_embeds = []
    for texts_batch in tqdm(texts_batches, desc="Computing caption embeddings", total=len(texts_batches)):
        texts_batch_inputs = siglip_tokenizer(
            texts_batch,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        batch_caption_embeds = clip.get_text_features(**texts_batch_inputs)
        batch_caption_embeds = F.normalize(batch_caption_embeds, dim=-1)
        caption_embeds.append(batch_caption_embeds.cpu())
    caption_embeds = torch.cat(caption_embeds, dim=0)
    return caption_embeds


def get_embeddings_top_k(caption_embeds: torch.FloatTensor, image_embeds: torch.FloatTensor,
                         k: int) -> torch.FloatTensor:
    """
    Get image indices to rerank for each caption.
    """
    scores = caption_embeds @ image_embeds.T
    top_k_indices = torch.topk(scores, k=k, dim=1).indices
    return top_k_indices


def get_embeddings_top_k_i2t(image_embeds: torch.FloatTensor, caption_embeds: torch.FloatTensor,
                              k: int) -> torch.FloatTensor:
    """
    Get caption indices to rerank for each image (I2T).
    """
    scores = image_embeds @ caption_embeds.T
    top_k_indices = torch.topk(scores, k=k, dim=1).indices
    return top_k_indices


@torch.inference_mode()
def evaluation_i2t(model: MultimodalPretrainModel, test_dataset: Dataset, device: torch.device, config: dict):
    siglip_path = config['siglip_path']
    clip = SiglipModel.from_pretrained(siglip_path).eval().to(device)
    siglip_tokenizer = AutoTokenizer.from_pretrained(siglip_path)

    image_embeds, image_token_embeds = compute_image_embeds(clip, test_dataset, device, config)
    caption_embeds = compute_caption_embeds(clip, siglip_tokenizer, test_dataset, device, config)
    top_k_indices = get_embeddings_top_k_i2t(image_embeds, caption_embeds, config['k'])

    clip_metrics = {f"R@{k}": 0 for k in K_LIST}
    reranked_metrics = {f"R@{k}": 0 for k in K_LIST}

    pbar = tqdm(total=len(test_dataset.image), desc="Evaluating I2T")

    for idx, image_path in enumerate(test_dataset.image):
        target_caption_idxs = test_dataset.img2txt[idx]

        clip_top_k_caption_idxs = top_k_indices[idx]
        
        # Get vision features for current image
        current_image_vision_features = image_token_embeds[idx:idx+1]  # Keep batch dimension
        
        # Get captions to rerank
        captions_to_rerank = [test_dataset.text[cap_idx] for cap_idx in clip_top_k_caption_idxs]
        
        scores = model.compute_logits(captions=captions_to_rerank,
                                      vision_encoder_output_tokens=current_image_vision_features.expand(
                                          len(captions_to_rerank), -1,
                                          -1).to(device))
        
        reranked_top_k_caption_idxs = [
            caption_idx for
            caption_idx, _ in
            sorted(zip(clip_top_k_caption_idxs, scores), key=lambda x: x[1], reverse=True)
        ]

        for k in K_LIST:
            clip_metrics[f"R@{k}"] += int(any(target_idx in clip_top_k_caption_idxs[:k] for target_idx in target_caption_idxs))
            reranked_metrics[f"R@{k}"] += int(any(target_idx in reranked_top_k_caption_idxs[:k] for target_idx in target_caption_idxs))

        pbar.set_description(
            f"CLIP R@1: {clip_metrics['R@1'] / (idx + 1):.4f} | Reranked R@1: {reranked_metrics['R@1'] / (idx + 1):.4f}")
        pbar.update(1)

    num_samples = len(test_dataset.image)
    clip_metrics = {k: v / num_samples for k, v in clip_metrics.items()}
    reranked_metrics = {k: v / num_samples for k, v in reranked_metrics.items()}
    print(f"I2T CLIP Metrics: {clip_metrics}")
    print(f"I2T Reranked Metrics: {reranked_metrics}")
    return reranked_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval.yaml')
    parser.add_argument('--checkpoint', default='/gfs/shared/shahaf/checkpoint_16.pth')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config_yaml = YAML(typ="safe")
    config = config_yaml.load(open(args.config, 'r'))

    model = MultimodalPretrainModel(
        siglip_path=config['siglip_path'],
        base_language_model_path=config['language_model_path'],
        multimodal_projection_hidden_dim=config['multimodal_projection_hidden_dim'],
        num_negatives_per_sample=config['num_negatives_per_sample'],
    )
    model_sd = torch.load(args.checkpoint)["model"]
    model.load_state_dict(model_sd)

    image_processor = AutoImageProcessor.from_pretrained(config['siglip_path'])
    siglip_transform = lambda x: image_processor(images=x, return_tensors='pt')['pixel_values'][0]
    
    # Choose dataset based on config
    test_set = config.get('test_set', 'flickr')  # Default to flickr if not specified
    if test_set.lower() == 'coco':
        test_dataset = coco_karpathy_retrieval_eval(siglip_transform, config['image_root'], config['ann_root'], 'test')
        print("Using COCO test dataset")
    else:
        test_dataset = flickr30k_retrieval_eval(siglip_transform, config['flickr_image_root'], config['flickr_ann_root'], 'test')
        print("Using Flickr30k test dataset")

    device = torch.device(args.device)

    model = model.to(device).eval()
    print("=== Text-to-Image Retrieval ===")
    t2i_metrics = evaluation_t2i(model, test_dataset, device, config)
    print("\n=== Image-to-Text Retrieval ===")
    i2t_metrics = evaluation_i2t(model, test_dataset, device, config)

    print("\n=== Final Results ===")
    print(f"T2I: {t2i_metrics}")
    print(f"I2T: {i2t_metrics}")
