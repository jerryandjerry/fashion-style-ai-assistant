from fashion_style_ai_assistant.intent import parse_intent


def test_parse_budget_and_beach_intent():
    intent = parse_intent("relaxed light outfit for beach weekend under 90 dollars")
    assert intent.budget_max == 90.0
    assert "beach" in intent.occasion_terms
    assert "casual" in intent.style_terms
    assert not intent.needs_clarification


def test_parse_chinese_terms():
    intent = parse_intent("面试穿的黑色正式外套，预算不超过 150")
    assert "interview" in intent.occasion_terms
    assert "black" in intent.color_terms
    assert "formal" in intent.style_terms
    assert intent.budget_max == 150.0
