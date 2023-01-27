import json

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


stoplist = []
with open('cat_stoplist.txt') as f:
    for line in f:
        stoplist.append(line)

unique_cats = set()

with open('data/cleaned.jsonl', 'w') as f:
    with open('crawler/items.jsonl', 'r') as s:
        for line in s:
            item = json.loads(line)
            if 'image' not in item or 'categories' not in item:
                continue
            cleaned_cats = []
            for cat in item['categories'].split(","):
                new_cat = cat.replace("Rezepte", "")
                new_cat = new_cat.strip()
                new_cat = convert_umlauts(new_cat)
                new_cat = new_cat.lower()
                # TODO: clean other special characters
                if new_cat not in stoplist:
                    unique_cats.add(new_cat)
                    cleaned_cats.append(new_cat)
            if len(cleaned_cats) <= 0:
                continue
            item['categories'] = cleaned_cats
            f.write(json.dumps(item) + "\n")

print("unique", len(unique_cats))

with open('data/unique_cats.txt', 'w') as f:
    for cat in unique_cats:
        f.write(cat + "\n")
