import os
import glob
import json
import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Aggregate metadata files.")
    parser.add_argument("--meta_folder", type=str, help="Path to the folder containing metadata (JSON files with image URLs and captions)")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--keyword_file", type=str, help="Path to the file containing keywords")
    
    return parser.parse_args()


def aggregate_meta_files(meta_folder: str, output_file: str, keyword_file: str) -> None:
    """
    Aggregate metadata files.
    Args:
    - meta_folder: str, path to the folder containing metadata (JSON files with image URLs and captions)
    - output_file: str, path to the output file
    """
    class_names = json.load(open(keyword_file, "r"))
    """
    Create a list of dictionaries, where each dictionary contains the metadata for a single image.
    The dictionary should have the following keys:
    - uuid: str, the unique identifier for the image
    - url: str, the URL of the image
    - caption: str, the caption for the image
    - class_id: str, the class ID for the image
    - class_name: str, the class name for the image
    - image_name: str, the name of the image file
    - abs_path: str, the absolute path to the image file
    - clip_score: float, the CLIP similarity score for the image
    """
    
    meta_files = glob.glob(f"{meta_folder}/*.json")
    meta_list = []
    for meta_file in meta_files:
        metadata = json.load(open(meta_file, "r"))
        for item in metadata:
            class_id = item["class_id"]
            class_name = class_names[class_id]
            image_ext = os.path.splitext(item["url"])[1] or ".jpg"
            image_name = f"{item['uuid']}{image_ext}"
            abs_path = os.path.join(meta_folder, "images", image_name)
            clip_score = item["similarity_score"]
            meta_list.append({
                "uuid": item["uuid"],
                "url": item["url"],
                "caption": item["caption"],
                "class_id": class_id,
                "class_name": class_name,
                "image_name": image_name,
                "abs_path": abs_path,
                "clip_score": clip_score
            })
            
    # Save the metadata
    ext = os.path.splitext(output_file)[1]
    match ext:
        case ".json":
            json.dump(meta_list, open(output_file, "w"), indent=4)
        case ".csv":
            df = pd.DataFrame(meta_list)
            df.to_csv(output_file, index=False)
        case ".h5" | ".hdf5":
            df = pd.DataFrame(meta_list)
            df.to_hdf(output_file, key="metadata", mode="w")
        case _:
            raise ValueError("Output file format not supported. Please use .json or .csv.")


if __name__ == "__main__":
    args = get_args()
    
    aggregate_meta_files(args.meta_folder, args.output_file, args.keyword_file)
    
