import os
import json
import argparse

import boto3
from utils.cc_matching import process

def list_all_wat_files(bucket: str, archive_prefix: str, save_uri_to_json: bool = False, json_path: str = None):
    s3 = boto3.client('s3')
    segments_prefix = f'crawl-data/{archive_prefix}/segments/'
    wat_files_uris = []

    # List all segments
    segments_response = s3.list_objects_v2(Bucket=bucket, Prefix=segments_prefix, Delimiter='/')
    if 'CommonPrefixes' not in segments_response:
        print("No segments found.")
        return

    for segment in segments_response['CommonPrefixes']:
        segment_prefix = segment['Prefix']
        wat_prefix = f'{segment_prefix}wat/'

        # Recursively list all .wat files in each segment's wat folder
        wat_files_response = s3.list_objects_v2(Bucket=bucket, Prefix=wat_prefix)
        while wat_files_response:
            if 'Contents' in wat_files_response:
                for item in wat_files_response['Contents']:
                    # Construct the full S3 URI for each .wat.gz file
                    wat_file_uri = f's3://{bucket}/{item["Key"]}'
                    wat_files_uris.append(wat_file_uri)

            if wat_files_response['IsTruncated']:
                wat_files_response = s3.list_objects_v2(Bucket=bucket, Prefix=wat_prefix, ContinuationToken=wat_files_response['NextContinuationToken'])
            else:
                break
            
    if not save_uri_to_json:
        return wat_files_uris
    
    output_folder = os.path.dirname(json_path)
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(wat_files_uris, f, indent=4)

def process_cc_archive(data_uri: str, folder: str, s3_client: boto3.client, metadata: str, temp_folder: str = 'temp_data/cc/temp_file/', bucket_name: str = 'commoncrawl'):
    os.makedirs(temp_folder, exist_ok=True)
    # download the file from s3
    print(f'Downloading {data_uri} from s3...')
    # get the prefix for s3 from the uri, e.g., remove s3://commoncrawl/
    s3_prefix = data_uri.split('commoncrawl/')[1]
    local_file_path = os.path.join(temp_folder, os.path.basename(data_uri))
    # check if the file already exists
    if os.path.exists(local_file_path):
        print(f'{local_file_path} already exists, proceed with the existing file.')
    else:
        s3_client.download_file(bucket_name, s3_prefix, local_file_path)
    # process and save the output
    print(f'Processing {local_file_path}...')
    output_file = os.path.join(folder, os.path.basename(data_uri).replace('.wat.gz', '.json'))
    process(local_file_path, output_file, metadata)
    # remove temp file
    print(f'Removing {local_file_path}...')
    os.remove(local_file_path)
    

def get_args():
    parser = argparse.ArgumentParser(description='Common Crawl Query')
    parser.add_argument('--output_folder', type=str, default='temp_data/cc/cc_matches', help='Output folder')
    parser.add_argument('--keyword_json', type=str, default='temp_data/dog_metadata.json', help='Metadata file')
    parser.add_argument('--archive_folder', type=str, default='temp_data/cc', help='Archive folder')
    parser.add_argument('--crawl', type=str, default='CC-MAIN-2023-50', help='Crawl name, e.g., CC-MAIN-2023-50')
    parser.add_argument('--bucket', type=str, default='commoncrawl', help='Bucket name')
    parser.add_argument('--num_files', type=int, default=None, help='Number of files to process')
    parser.add_argument('--format', choices=['wat', 'warc'], default='wat', help='Format of the meta files')
    
    
    # AWS credientials
    parser.add_argument('--aws_access_key_id', type=str, default=None, help='AWS access key id')
    parser.add_argument('--aws_secret_access_key', type=str, default=None, help='AWS secret access key')
    parser.add_argument('--aws_session_token', type=str, default=None, help='AWS session token')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    if not os.path.exists(f'{args.archive_folder}/{args.archive}/{args.format}_file_uris.json'):
        print(f"Getting {args.format} file uris from {args.bucket}/{args.crawl} and save to {args.archive_folder}/{args.crawl}/{args.format}_file_uris.json...")
        list_all_wat_files(args.bucket, args.crawl, save_uri_to_json=True, json_path=f'{args.archive_folder}/{args.crawl}/{args.format}_file_uris.json')
    # read the wat file uris
    wat_file_uris = json.load(open(f'{args.archive_folder}/{args.crawl}/{args.format}_file_uris.json', 'r'))

    if args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        s3_client = boto3.client('s3', 
                                 region_name='us-east-1', 
                                 aws_access_key_id=args.aws_access_key_id, 
                                 aws_secret_access_key=args.aws_secret_access_key, 
                                 aws_session_token=args.aws_session_token)
    else:
        s3_client = boto3.client('s3', region_name='us-east-1')

    process_file_count = 0
    for wat_file_uri in wat_file_uris:
        if args.num_files is not None and process_file_count >= args.num_files:
            print(f"Processed {args.num_files} files, stop processing.")
            break
        # check if exist
        output_file = os.path.join(args.output_folder, os.path.basename(wat_file_uri).replace('.wat.gz', '.json'))
        if os.path.exists(output_file):
            print(f'{output_file} already exists.')
            continue
        process_cc_archive(wat_file_uri, args.output_folder, s3_client, args.keyword_json)
        process_file_count += 1
