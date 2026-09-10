from eval.runner import evaluate_assistant

from fashion_style_ai_assistant.data import _last_purchase_holdout, load_catalog_bundle
from fashion_style_ai_assistant.baseline.pipeline import FashionAssistant
from fashion_style_ai_assistant.schemas import CatalogBundle


def _run(**kwargs):
    """Compose the eval the way the CLI does: prep data, build the examinee, score it."""
    bundle = load_catalog_bundle("data/sample")
    train, test = _last_purchase_holdout(bundle)
    assistant = FashionAssistant(
        CatalogBundle(
            products=bundle.products,
            interactions=train,
            reviews=bundle.reviews,
            inventory=bundle.inventory,
            users=bundle.users,
            outfits=bundle.outfits,
            data_dir=bundle.data_dir,
        )
    )
    return evaluate_assistant(
        assistant, test, bundle.products, bundle.inventory, top_k=5, **kwargs
    )


def test_evaluate_smoke():
    metrics = _run()
    assert metrics["users_evaluated"] > 0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0


def test_evaluate_can_sample_users():
    metrics = _run(sample_users=2, sample_seed=7)
    assert metrics["users_evaluated"] == 2.0


def test_evaluate_sample_users_above_available_uses_all_users():
    all_metrics = _run()
    sampled_metrics = _run(sample_users=999)
    assert sampled_metrics["users_evaluated"] == all_metrics["users_evaluated"]
