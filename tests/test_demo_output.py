from fashion_style_ai_assistant.cli import main

SAMPLE_QUERY = "I need a relaxed light outfit for a beach weekend under 90 dollars"


def test_demo_end_to_end_markdown(tmp_path):
    """End-to-end: drive the CLI demo and verify the generated markdown structure.

    Writes to a temp path (not docs/) so the test has no repo side effects.
    Regenerating the docs/ sample is the job of scripts/run_demo.py.
    """
    out_path = tmp_path / "result.md"
    main(
        [
            "--config", "config.yaml",
            "demo",
            "--user-id", "u001",
            "--top-k", "1",
            "--query", SAMPLE_QUERY,
            "--output", str(out_path),
        ]
    )

    text = out_path.read_text(encoding="utf-8")
    # header
    assert text.startswith("# Recommendation result")
    assert "`u001`" in text
    assert f"**Query:** `{SAMPLE_QUERY}`" in text
    # recommendation + grounded, aggregated review evidence
    assert "1. Linen Relaxed Shirt" in text
    assert "Reviews: avg" in text                    # review aggregation rendered
    assert "Sample top reviews:" in text             # dedicated top-reviews section
    assert "r001.review_text" in text                # sampled quote listed by review id
    assert "Very breathable and relaxed" in text     # verbatim grounded quote present
    # business sections
    assert "## Business metrics" in text
    assert "## Business interpretation" in text
