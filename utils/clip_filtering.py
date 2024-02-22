import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

class DatasetFromJson(Dataset):
    def __init__(self, json_path: str, image_folder: str, transform: callable = None):
        """
        Args:
        - json_path: str, path to the json file containing the metadata
        - image_folder: str, path to the folder containing images
        - transform: callable, a function that takes in an image and returns a transformed version
        """
        self.json_path = json_path
        self.image_folder = image_folder
        self.transform = transform
        
        # Construct the data from the json file
        self._construct_data_from_json(json_path, image_folder)

    def _construct_data_from_json(self, json_path: str, image_folder: str) -> None:
        self.metadata = json.load(open(json_path, "r"))
        self.samples = []
        self.image_ids = []
        self.targets = []
        for item in self.metadata:
            image_id = item['uuid']
            image_ext = os.path.splitext(item['url'])[1] or '.jpg'
            image_path = os.path.join(image_folder, f"{image_id}{image_ext}")
            class_id = item['texts'][0][2][0]
            self.samples.append((image_path, class_id))
            self.image_ids.append(image_id)
            self.targets.append(class_id)
    
    def __getitem__(self, index: int) -> tuple:
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target
            
            
def get_clip_scores(model: torch.nn.Module,
                    transforms: callable,
                    keywords: list[str] | str,
                    image_folder: str,
                    json_path: str,
                    device: str = "cuda:0"
                    ) -> dict:
    """
    Based on the input json file, this function will compute the (average) similarity scores between the images and the keywords.
    Args:
    - model: torch.nn.Module, CLIP model
    - keywords: list[str] | str, list of keywords or a single keyword
    - image_folder: str, path to the folder containing images
    - json_path: str, path to the json file containing the metadata
    Returns:
    - dict, the original metadata with the similarity scores added
    """
    
    print(f"Using device: {device}, model: {model}, image_folder: {image_folder}, json_path: {json_path}")
    print(f"Getting similarity scores for {keywords}...")
    dataset = DatasetFromJson(json_path, image_folder, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=12, pin_memory=True, drop_last=False)
    
    text_features = model.encode_text(keywords).to(device)
    if isinstance(keywords, list):
        text_features = text_features.mean(dim=0, keepdim=True)
    
    sim_scores = []
    for sample, target in dataloader:
        with torch.no_grad():
            image_features = model.encode_image(sample.to(device))
            # compute cosine similarity between image and text features
            sim_score = (image_features @ text_features.T).squeeze().cpu().tolist()
            sim_scores.extend(sim_score)
    
    # insert socres into metadata
    for i, item in enumerate(dataset.metadata):
        item["clip_score"] = sim_scores[i]
    
    return dataset.metadata