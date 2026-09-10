# Data directory

- `sample/`: tiny synthetic data that makes the project runnable.
- `external/`: place downloaded public datasets here. This directory is gitignored.
- `processed/`: generated normalized datasets. This directory is gitignored.

Expected normalized schema:

## products.csv

Required columns:

- `product_id`
- `name`
- `category`
- `color`
- `style_tags`
- `occasion_tags`
- `season`
- `price`
- `description`

Optional but useful columns:

- `brand`
- `material`
- `gender`
- `image_url`

## interactions.csv

Required columns:

- `user_id`
- `product_id`
- `event_type`
- `event_timestamp`

Optional:

- `event_weight`

## reviews.csv

Required columns:

- `review_id`
- `product_id`
- `rating`
- `fit_feedback`
- `review_text`

## inventory.csv

Required columns:

- `product_id`
- `stock`
- `margin`
- `return_rate`

## outfits.csv

Required columns:

- `outfit_id`
- `product_ids` as pipe-separated IDs, e.g. `p001|p002|p003`
- `occasion_tags`
- `description`
- `compatibility_score`
