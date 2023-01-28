import json

import re


def convert_umlauts(content):
    # pre-processing: convert umlauts
    umlaut_mapping = {
        u'Ä': 'Ae',
        u'Ö': 'Oe',
        u'Ü': 'Ue',
        u'ä': 'ae',
        u'ö': 'oe',
        u'ü': 'ue',
        u'ß': 'ss',
    }
    content = content.translate({ord(key): val for key, val in umlaut_mapping.items()})
    return content


def clean_cat(new_cat):
    new_cat = convert_umlauts(new_cat)
    new_cat = new_cat.lower()
    new_cat = re.sub(r'\d+', ' ', new_cat)
    new_cat = re.sub(r'[^A-Za-z0-9]+', ' ', new_cat)
    new_cat = new_cat.strip()
    return new_cat


def get_mapping(cat):
    if cat in mapping:
        return mapping[cat]
    return []


def invert_mapping(map):
    new_map = {}
    for key, vals in map.items():
        for val in vals:
            key = clean_cat(key)
            val = clean_cat(val)
            if val not in new_map:
                new_map[val] = []
            new_map[val].append(key)
    return new_map


#
# stoplist = []
# with open('cat_stoplist.txt') as f:
#     for line in f:
#         line = clean_cat(line)
#         if len(line) > 2:
#             stoplist.append(line)

with open('new_categories.json') as f:
    mapping = json.load(f)

mapping = invert_mapping(mapping)

unique_cats = set()

with open('data/cleaned.jsonl', 'w') as f:
    with open('crawler/items.jsonl', 'r') as s:
        for line in s:
            item = json.loads(line)
            if 'image' not in item or 'categories' not in item:
                continue
            cleaned_cats = set()
            for cat in item['categories'].split(","):
                new_cat = cat.replace("Rezepte", "")
                new_cat = clean_cat(new_cat)
                new_cats = get_mapping(new_cat)
                for ncat in new_cats:
                    cleaned_cats.add(ncat)
                    unique_cats.add(ncat)
            if len(cleaned_cats) <= 0:
                continue
            out = {'categories': list(cleaned_cats), 'image': item['image']}
            f.write(json.dumps(out) + "\n")

print("unique", len(unique_cats))

with open('data/unique_cats.txt', 'w') as f:
    for cat in unique_cats:
        f.write(cat + "\n")
