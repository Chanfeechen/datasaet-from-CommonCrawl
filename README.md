# Construct image dataset from Common Crawl data
This repo provide easy scripts to construct image dataset based on keywords from Common Crawl data. The dataset is constructed by downloading images from the webpages that contain the keywords. The images are filtered based on OPENAI CLIP model.

## Prerequisites
1. Install Pytorch from the official website: https://pytorch.org/get-started/locally/
e.g., on Linux, you can install Pytorch by running the following command:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. Install other required packages by running the following command:
```bash
pip install -r requirements.txt
```
(Make sure Python >= 3.10)

## Usage
### Step1: Specify a crawl
You can find crawls at [Common Craw website](https://commoncrawl.org/overview). e.g., `CC-MAIN-2023-50`.

### Step2: Setup AWS credentials (Optional, recommanded)
You need to setup AWS credentials to download the crawl data. You can find the command line instructions at [AWS website](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html).

### Step3: Query the crawl
You can query the crawl by running the following command:
```bash
python query_crawl.py --crawl CC-MAIN-2023-50 --metadata_file /path/to/ketword/json --archive_folder /path/to/output --output_folder /path/to/output
```
- `--crawl`: the crawl id. e.g., `CC-MAIN-2023-50`.
- `--keyword_json`: the path to the json file that contains the keywords. e.g., `keywords.json`.
- `--archive_folder`: the path to the folder to store the temporary data downloaded crawl data. e.g., `temp_data/cc`.
- `--output_folder`: the path to the folder to store the output json which contains the matched items and image url. e.g., `output_data/cc_matches`.

If you did not setup AWS credentials, you can specify the credientials by using the following arguments:
- `--aws_access_key_id`: the access key id.
- `--aws_secret_access_key`: the secret access key.
- `--aws_session_token`: the session token.

### Step4: Download images
You can download images by running the following command:
```bash
python download_images.py --meta_folder /path/to/matches/json --output_folder /path/for/output/images --num_workers 25 --keyword_json /path/to/keyword/json
```

### Step5: Filter images based on CLIP model
You can filter images based on CLIP model by running the following command:
```bash
python filter_images.py --image_folder /path/to/images --meta_folder /path/to/matches/json --model_name "ViT-B/32" --device "cuda:0" --output_folder /path/to/output --threshold 27.5 --delete_images
```

The threshold can be adjusted based on the specific task. It is recommanded to use a desired subset of data to find the threshold. The `delete_images` argument is used to delete the images that do not pass the threshold.