from src.bonus.risk_warning_pipeline import find_keyword_evidence, find_opinion_evidence
from src.llm.risk_warnings import extract_audit_opinion


def test_extract_audit_opinion_qualified():
    text = "Our report contains a qualified opinion due to scope limitation."
    result = extract_audit_opinion(text)
    assert result["opinion"] == "qualified"


def test_find_keyword_evidence():
    pages = ["Liquidity risk is high due to refinancing pressure."]
    evidence = find_keyword_evidence(pages, ["liquidity"], max_hits=1)
    assert evidence
    assert evidence[0]["keyword"] == "liquidity"


def test_find_opinion_evidence():
    pages = ["DISCLAIMER OF OPINION was issued by the auditor."]
    evidence = find_opinion_evidence(pages, max_hits=1)
    assert evidence
    assert evidence[0]["pattern"] == "disclaimer"
