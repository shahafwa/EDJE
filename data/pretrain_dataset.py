import json

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform):

        self.ann_pretrain = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann_pretrain += ann

        self.annotation = self.ann_pretrain
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        image_path, caption = ann['image'], ann['caption']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(caption)

        return image, caption
