# IUI

## Setup

```bash
# Setup Python Env
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Run crawler and pre-processing
```bash
cd crawler
# Crawl data
scrapy crawl gutekueche
# Process data
python3 pre_process.py
# Download images
python3 get_images.py
# Split into test/train
bash split.sh
```

## Train model

```bash
python3 train_model.py
```

## Run Server

```bash
# Run Server
flask run 

# Run Server in debug mode
flask --debug run
```

## Testing

```bash
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/pred -F "file=@data/demo/cookie.jpg"
```