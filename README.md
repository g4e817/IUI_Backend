# IUI

## Setup

```bash
# Setup Python Env
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Run crawler
```bash
cd crawler
scrapy crawl gutekueche
python3 pre_process.py
python3 get_images.py
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