from fashion_style_ai_assistant.baseline.pipeline import FashionAssistant


def test_pipeline_recommendations_have_evidence():
    assistant = FashionAssistant.from_data_dir("data/sample")
    response = assistant.recommend(
        query="I need a relaxed light outfit for a beach weekend under 90 dollars",
        user_id="u001",
        top_k=3,
    )
    assert len(response["recommendations"]) == 3
    assert response["recommendations"][0]["evidence"]
    assert response["business_metrics"]["num_recommendations"] == 3
