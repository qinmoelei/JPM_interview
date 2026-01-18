from src.bonus.credit_rating_pipeline import collect_tickers, find_opinion_evidence, rating_override


def test_collect_tickers(tmp_path):
    (tmp_path / "AAA_states.csv").write_text("x")
    (tmp_path / "BBB_states.csv").write_text("x")
    tickers = collect_tickers(tmp_path)
    assert tickers == ["AAA", "BBB"]


def test_rating_override_rules():
    assert rating_override("disclaimer", {}) == "D"
    assert rating_override("adverse", {}) == "D"
    assert rating_override("qualified", {}) == "CCC"
    assert rating_override("unqualified", {"going_concern": True}) == "CC"
    assert rating_override("unqualified", {}) is None


def test_find_opinion_evidence_disclaimer():
    pages = ["This is a DISCLAIMER OF OPINION in the audit report."]
    evidence = find_opinion_evidence(pages, max_hits=1)
    assert evidence
    assert evidence[0]["pattern"] == "disclaimer"
