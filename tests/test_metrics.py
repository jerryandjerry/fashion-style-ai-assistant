from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k


def test_ranking_metrics():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert average_precision_at_k(recommended, relevant, 4) > 0
    assert ndcg_at_k(recommended, relevant, 4) > 0
