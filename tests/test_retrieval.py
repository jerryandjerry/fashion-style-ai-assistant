from fashion_style_ai_assistant.data import load_catalog_bundle
from fashion_style_ai_assistant.intent import parse_intent
from fashion_style_ai_assistant.baseline.retrieval import TfidfCatalogRetriever


def test_retriever_returns_beach_items():
    bundle = load_catalog_bundle("data/sample")
    retriever = TfidfCatalogRetriever(bundle.products)
    results = retriever.search(parse_intent("beach linen relaxed outfit"), top_k=5)
    assert len(results) == 5
    assert results.iloc[0]["product_id"] in {"p001", "p002", "p003", "p019"}
