from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data import make_product_text
from ..schemas import ParsedIntent


class TfidfCatalogRetriever:
    """Offline vector-style retriever using TF-IDF.

    This class intentionally avoids external services so the project is reproducible offline.
    It exposes the same shape you would expect from a vector database search result.
    """

    def __init__(self, products: pd.DataFrame, max_features: int = 5000):
        if products.empty:
            raise ValueError("products cannot be empty")
        self.products = products.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
        self.search_text = make_product_text(self.products)
        '''
                         linen   relaxed   shirt   ivory   beach   shoes   ...
        product p001      0.37    0.25     0.34    0.28    0.10    0.00
        product p002      0.22    0.00     0.00    0.00    0.08    0.00
        product p003      0.00    0.00     0.00    0.00    0.09    0.00
        ...
        '''
        self.matrix = self.vectorizer.fit_transform(self.search_text)

    
    def search(self, parsed_intent: ParsedIntent, top_k: int = 30) -> pd.DataFrame:
        query_text = parsed_intent.as_retrieval_text()
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        candidates = self.products.copy()
        candidates["retrieval_score"] = scores.astype(float)

        if parsed_intent.budget_max is not None:
            # Soft price filtering: keep items within budget and near-budget items, but down-rank later.
            candidates = candidates[candidates["price"] <= parsed_intent.budget_max * 1.25].copy()

        if candidates.empty:
            candidates = self.products.copy()
            candidates["retrieval_score"] = scores.astype(float)

        candidates = candidates.sort_values(["retrieval_score", "price"], ascending=[False, True])
        candidates["retrieval_rank"] = np.arange(1, len(candidates) + 1)
        return candidates.head(top_k).reset_index(drop=True)
