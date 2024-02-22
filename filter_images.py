import os
import glob
import json
import argparse

import clip
from tqdm import tqdm

from utils.clip_filtering import get_clip_scores

def get_args():
    parser = argparse.ArgumentParser(description="Filter images based on CLIP.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--meta_folder", type=str, help="Path to the folder containing metadata (JSON files with image URLs and captions)")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="Name of the CLIP model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for CLIP")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where filtered meta will be saved")
    parser.add_argument("--threshold", type=float, default=27.5, help="Threshold for similarity score, for ViT-B/32, suggested values are between 27 and 30")
    parser.add_argument("--delete_images", action="store_true", help="Delete images that are under the threshold")
    
    return parser.parse_args()

def filter_images_with_clip(model_name: str,
                            image_folder: str,
                            meta_folder: str,
                            output_folder: str,
                            device: str = "cuda:0",
                            threshold: float = 27.5,
                            delete_images: bool = False
                            ) -> None:
    """
    Filter images based on CLIP similarity score.
    Args:
    - model_name: str, name of the CLIP model
    - image_folder: str, path to the folder containing images
    - meta_folder: str, path to the folder containing metadata (JSON files with image URLs and captions)
    - output_folder: str, path to the folder where filtered meta will be saved
    - device: str, device to use for CLIP
    - threshold: float, threshold for similarity score, for ViT-B/32, suggested values are between 27 and 30
    - delete_images: bool, delete images that are under the threshold
    """
    
    # Load the model
    model, preprocess = clip.load(model_name, device)

    # Get the list of JSON files
    json_files = glob.glob(f"{meta_folder}/*.json")
    
    for json_file in tqdm(json_files, desc="Filtering images", total=len(json_files)):
        new_meta = []
        metadata = get_clip_scores(model, preprocess, json_file, image_folder, device)
        for item in metadata:
            if item["similarity_score"] < threshold:
                if delete_images:
                    os.remove(item["image_path"])
                else:
                    item["delete"] = True
            new_meta.append(item)
        # Save the new metadata
        output_file = os.path.join(output_folder, os.path.basename(json_file))
        json.dump(new_meta, open(output_file, "w"), indent=4)

if __name__ == "__main__":
    args = get_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    filter_images_with_clip(model_name=args.model_name,
                            image_folder=args.image_folder,
                            meta_folder=args.meta_folder,
                            output_folder=args.output_folder,
                            device=args.device,
                            threshold=args.threshold,
                            delete_images=args.delete_images)