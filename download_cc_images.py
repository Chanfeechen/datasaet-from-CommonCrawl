import os
import json
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

def download_image_for_item(item, output_folder, verbose=False):
    uuid = item["uuid"]
    url = item["url"]
    if len(item["texts"][0]) >= 3:
        caption = item["texts"][0][1]
        class_id = item["texts"][0][2]
    ext = extract_extension(url)
    output_path = os.path.join(output_folder, f"{uuid}{ext}")
    # if image already exists, skip
    if os.path.exists(output_path):
        if verbose:
            print(f"Image already exists: {output_path}")
        return (uuid, url, caption, class_id)
    return (uuid, url, caption, class_id) if download_image(url, output_path, verbose) else None

def process_json_file(json_path, output_folder, workers=4, verbose=False):
    with open(json_path, 'r') as file:
        data = json.load(file)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit a task for each item
        futures = [executor.submit(download_image_for_item, item, output_folder, verbose) for item in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Downloading images from {os.path.basename(json_path)}"):
            if result := future.result():
                results.append(result)

    return results

def download_image(image_url: str, output_path: str, verbose: bool = False, timeout: int = 5):
    try:
        response = requests.get(image_url, stream=True, timeout=timeout)
        if response.status_code == 200:
            with open(output_path, 'wb') as out_file:
                for chunk in response.iter_content(1024):
                    out_file.write(chunk)
            return True
    except requests.exceptions.Timeout:
        if verbose:
            print(f"Request timed out for {image_url}")
    except Exception as e:
        if verbose:
            print(f"Error downloading {image_url}: {e}")
    return False

def extract_extension(url):
    parsed_url = urlparse(url)
    root, ext = os.path.splitext(parsed_url.path)
    return ext or '.jpg'  # Default to .jpg if no extension found

def create_metadata_file(results: dict, output_path: str, clas_list: list):
    return [
        {
            "uuid": item[0],
            "url": item[1],
            "caption": item[2],
            "class_id": item[3],
        }
        for item in results
    ]

def process_all_json_files(folder_path: str, output_folder: str, meta_output_folder: str, class_list: list[str], workers: int = 25):
    total_images = 0
    for file_name in tqdm(os.listdir(folder_path), desc="Overall progress", total=len(os.listdir(folder_path))):
        if file_name.endswith('.json'):
            json_path = os.path.join(folder_path, file_name)
            output_meta_path = os.path.join(meta_output_folder, f"{file_name.replace('.json', '_metadata.json')}")
            # if meta already exist, skip
            if os.path.exists(output_meta_path):
                continue
            results = process_json_file(json_path, output_folder, workers=workers)
            meta_file = create_metadata_file(results, meta_output_folder, class_list)
            json.dump(meta_file, open(output_meta_path, 'w'), indent=4)
            total_images += len(results)
    print(f"Total images downloaded: {total_images}")

def get_args():
    parser = argparse.ArgumentParser(description="Download images from a JSON file")
    parser.add_argument("--json_folder", help="Path to the folder containing JSON files")
    parser.add_argument("--output_folder", help="Path to the folder where images will be saved")
    parser.add_argument("--workers", type=int, default=25, help="Number of workers for parallel processing")
    parser.add_argument("--keyword_json", type=str, default="/home/lab/datasets/cc_dogs/query_keywords.json", help="Path to the metadata file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    class_list = json.load(open(args.keyword_json, 'r'))
    images_folder = os.path.join(args.output_folder, "images")
    meta_folder = os.path.join(args.output_folder, "metadata")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    
    process_all_json_files(args.json_folder, images_folder, meta_folder, class_list)