import json
import requests
import re

with open('data/cleaned.jsonl') as f:
    for line in f:
        item = json.loads(line)
        image_url = item['image']
        image_data = requests.get(image_url).content

        match = re.match(r'media/recipe/(\d+)/conv/', image_url)
        recipe_id = match.group(0)
        with open(f"data/images/{recipe_id}.jpg", 'wb') as i:
            i.write(image_data)
