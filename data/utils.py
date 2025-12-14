import re
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

import numpy as np
from PIL import Image
from open_clip.tokenizer import canonicalize_text, basic_clean
from transformers import AutoTokenizer, AutoImageProcessor, BaseImageProcessor, PreTrainedTokenizer
CLIP_PATH = "google/siglip2-base-patch16-224"


def pre_caption(caption):
    return canonicalize_text(basic_clean(caption))

class DummyTransform:
    def __init__(self, image_processor:Optional=None):
        self.image_processor = image_processor
    def __call__(self, image: Image.Image, caption: str, negative_images: Optional[List[Image.Image]]=None) -> Tuple[Image.Image, str]:
        negative_images_pxls = []
        image = self.image_processor(images=image, return_tensors='pt')['pixel_values'][0]
        caption = pre_caption(caption)

        if negative_images:
            for neg_image in negative_images:
                negative_images_pxls.append(self.image_processor(images=neg_image, return_tensors='pt')['pixel_values'][0])
            return image, caption, negative_images_pxls
        return image, caption


class ShardCache(ABC):

    def __init__(
            self,
            index: List[dict]
    ):
        self.index = index
        self._shard_sizes = [shard_metadata["size"] for shard_metadata in self.index]
        self._overall_size = sum(self._shard_sizes)
        self._cum_shard_sizes = np.insert(np.cumsum(self._shard_sizes), 0, 0)[:-1]

        self._cached_shard_index: Optional[int] = None
        self._cached_shard: Optional[Any] = None

    def get_sample(self, index: int):
        shard_index, inner_index = self._decode_index(index)
        if shard_index != self._cached_shard_index:
            self._load_shard_to_cache(shard_index)
        sample = self._get_sample_from_shard(self._cached_shard, inner_index)
        return sample

    @abstractmethod
    def _get_sample_from_shard(self, shard: Any, shard_inner_index: int) -> Any:
        pass

    def _decode_index(self, index: int) -> Tuple[int, int]:
        shard_index = np.searchsorted(self._cum_shard_sizes, index, side="right") - 1
        inner_index = index - int(self._cum_shard_sizes[shard_index])
        return shard_index, inner_index

    def _load_shard_to_cache(self, shard_index: int):
        shard_metadata = self.index[shard_index]
        shard = self._get_shard_from_metadata(shard_metadata)
        self._cached_shard_index = shard_index
        self._cached_shard = shard

    @abstractmethod
    def _get_shard_from_metadata(self, shard_metadata: dict) -> Any:
        pass

    def __len__(self):
        return self._overall_size
