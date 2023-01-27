import json
import os.path
import requests
from tqdm import tqdm

counter = 0
num_lines = sum(1 for line in open('data/cleaned.jsonl'))

with open('data/cleaned.jsonl') as f:
    for line in tqdm(f, total=num_lines):
        item = json.loads(line)
        image_url = item['image']
        filename = os.path.basename(image_url)
        filepath = f"data/images/{filename}"

        if os.path.isfile(filepath):
            # Skip already downloaded images
            continue

        image_data = requests.get(image_url).content
        with open(filepath, 'wb') as i:
            i.write(image_data)
