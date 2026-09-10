# Recommended public datasets

This project is designed to run on the tiny sample dataset in `data/sample/`, then scale to open or public datasets.

## 1. H&M Personalized Fashion Recommendations

- Link: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
- Best role: main recommendation dataset.
- Useful files: `articles.csv`, `customers.csv`, `transactions_train.csv`, article images.
- Good project use cases:
  - personalized next-item recommendation
  - time-based holdout evaluation
  - user and product feature engineering
  - candidate generation + ranking
- Suggested integration:
  ```bash
  mkdir -p data/external/hm
  # Download via Kaggle UI or kaggle CLI into data/external/hm
  python scripts/data_ingest/ingest_hm.py --raw-dir data/external/hm --out-dir data/processed/hm --sample-size 50000
  fashion-assistant demo --data-dir data/processed/hm --user-id <customer_id> --query "casual black jacket for autumn"
  ```
- Data info:
  - Normalized output: `products.csv`, `interactions.csv`, and `users.csv` come from H&M; 
  - `inventory.csv` is synthetic stock/margin/return-rate; 
  - `reviews.csv` and `outfits.csv` are empty schema files.

## 2. Polyvore Outfits

- Link: https://mariya.fyi/polyvore
- Alternate code/data page: https://github.com/mvasil/fashion-compatibility
- Best role: outfit compatibility and complete-the-look.
- Good project use cases:
  - outfit compatibility classifier
  - fill-in-the-blank item retrieval
  - bundle recommendation
  - multi-item explanation: “why these items work together”
- Suggested integration:
  ```bash
  mkdir -p data/external/polyvore
  python scripts/data_ingest/ingest_polyvore.py --raw-dir data/external/polyvore --out-dir data/processed/polyvore
  ```
- Data info:
  - Normalized output: `outfits.csv` and item-derived `products.csv` come from Polyvore; 
  - `interactions.csv`, `reviews.csv`, `inventory.csv`, and `users.csv` are empty schema files.

## 3. UCSD Clothing Fit Data: RentTheRunway / ModCloth

- Link: https://cseweb.ucsd.edu/~jmcauley/datasets.html
- Kaggle mirror: https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation
- Best role: fit-risk, size-risk, review-grounded explanations.
- Good project use cases:
  - classify `small` / `fit` / `large`
  - explain whether reviews mention “runs small”, “stretchy”, “tight waist”, etc.
  - estimate return-risk proxy
- Suggested integration:
  ```bash
  mkdir -p data/external/fit
  python scripts/data_ingest/ingest_fit_reviews.py \
    --reviews-json data/external/fit/renttherunway_final_data.json \
    --out-dir data/processed/fit
  ```
- Data info:
  - Normalized output: only `reviews.csv` comes from the fit dataset; 
  - product catalog, interactions, users, inventory, and outfits must come from another dataset.
- Caveat: user measurement fields such as privacy, fairness, age/body type are sensitive in the real practice.

## 4. Amazon Reviews 2023

- Link: https://amazon-reviews-2023.github.io/
- Hugging Face: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Best role: review RAG, item metadata, product search, bought-together graph.
- Good project use cases:
  - summarize common positives/negatives from reviews
  - product-to-product retrieval
  - cold-start product search using metadata
  - graph features from bought-together links
- Suggested integration:
  ```bash
  mkdir -p data/external/amazon2023
  python scripts/data_ingest/ingest_amazon_reviews.py \
    --reviews-jsonl data/external/amazon2023/Amazon_Fashion.jsonl \
    --meta-jsonl data/external/amazon2023/meta_Amazon_Fashion.jsonl \
    --out-dir data/processed/amazon_fashion
  ```
- Data info:
  - Normalized output: `products.csv`, `reviews.csv`, `interactions.csv`, and `users.csv` come from Amazon metadata/reviews; 
  - `inventory.csv` uses default stock/margin/return-rate; 
  - `outfits.csv` is an empty schema file.

## 5. Fashion Product Images Small / Myntra

- Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- Best role: quick image + metadata demo.
- Good project use cases:
  - image classification
  - text/image hybrid retrieval
  - “find visually similar products” with CLIP embeddings
- Suggested integration:
  ```bash
  mkdir -p data/external/fashion-product-images-small
  # Download Kaggle files into data/external/fashion-product-images-small
  python scripts/data_ingest/ingest_fashion_images.py \
    --raw-dir data/external/fashion-product-images-small \
    --out-dir data/processed/fashion_images
  ```
- Data info:
  - Normalized output: `products.csv` and `image_url` values come from `styles.csv` plus local images; 
  - `inventory.csv` uses default stock/margin/return-rate; 
  - `interactions.csv`, `reviews.csv`, `users.csv`, and `outfits.csv` are empty schema files.

## 6. BigQuery theLook eCommerce

- Link: https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce
- BigQuery public datasets docs: https://cloud.google.com/bigquery/public-data
- Best role: business KPI layer.
- Useful tables often include users, orders, order_items, products, inventory_items, events, and distribution centers.
- Good project use cases:
  - SQL funnel: product view → cart → purchase
  - inventory-aware recommendations
  - return/refund analysis
  - category margin and revenue proxy
- Suggested integration:
  ```bash
  mkdir -p data/external/thelook
  # Export BigQuery tables or download a CSV mirror into data/external/thelook
  python scripts/data_ingest/ingest_thelook.py \
    --raw-dir data/external/thelook \
    --out-dir data/processed/thelook \
    --sample-users 1000
  ```
- Data info:
  - Normalized output: `products.csv`, `users.csv`, `interactions.csv`, and `inventory.csv` come from theLook tables; 
  - `reviews.csv` and `outfits.csv` are empty schema files.
