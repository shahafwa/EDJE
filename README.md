# EDJE: Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking

[![arXiv](https://img.shields.io/badge/arXiv-2510.06820-b31b1b.svg)](https://arxiv.org/abs/2510.06820)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://shahafwa.github.io/EDJE/)

**[Mitchell Keren Taraday](https://arxiv.org/search/cs?searchtype=author&query=Taraday,+Mitchell+Keren)\*, [Shahaf Wagner](https://arxiv.org/search/cs?searchtype=author&query=Wagner,+Shahaf)\*, [Chaim Baskin](https://arxiv.org/search/cs?searchtype=author&query=Baskin,+Chaim)**
*INSIGHT Lab, Ben-Gurion University of the Negev, Israel*

This repository contains the official implementation of **EDJE**, a method for efficient large-scale vision-language reranking.

![Teaser](assets/teaser.png)

## Abstract
Multimodal retrieval still leans on embedding-based models like CLIP for fast vector search over pre-computed image embeddings. Yet, unlike text retrieval, where joint-encoder rerankers are standard, comparable vision-language rerankers are largely absent. We find that seminal joint encoders such as BLIP are severely bottlenecked by an expensive visual feature-extraction stage. EDJE introduces a lightweight attention-based adapter to compress precomputed vision tokens, enabling high-throughput inference (50k pairs/sec) while preserving strong retrieval performance.

## Installation

```bash
git clone https://github.com/shahafwa/EDJE.git
cd Simple-Efficient-Fusion
pip install -r requirements.txt
```

*(Note: Please ensure you have PyTorch and Transformers installed.)*

## Usage

### Training
To train the EDJE model:
```bash
python -m torch.distributed.run --nproc_per_node=8 train.py
```

### Evaluation
To evaluate on Flickr30k or MS-COCO:
```bash
python -m torch.distributed.run --nproc_per_node=8 evaluation/evaluate_retrieval.py --checkpoint /path/checkpoint.pth
```

## Project Website
Visit our [Project Website](https://gitanony04-lab.github.io/Simple-Efficient-Fusion/) for more visual results and details.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{taraday2025edje,
  title={Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking},
  author={Taraday, Mitchell Keren and Wagner, Shahaf and Baskin, Chaim},
  journal={arXiv preprint arXiv:2510.06820},
  year={2025}
}
```
