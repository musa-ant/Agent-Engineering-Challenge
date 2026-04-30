"""
evals/eval.py — generate + assert eval harness for the Helios RFP agent.

Generate (hits the API once per RFP, writes output/<rfp_id>.json):
    python -m evals.eval --generate

Assert (offline, fast, repeatable):
    pytest evals/eval.py -v
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_rfps"
OUTPUT_DIR = ROOT / "output"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sample_rfps() -> list[Path]:
    return sorted(SAMPLE_DIR.glob("*.json"))


def output_for(rfp_path: Path) -> Path:
    rfp = json.loads(rfp_path.read_text())
    return OUTPUT_DIR / f"{rfp['rfp_id']}.json"


# --------------------------------------------------------------------- generate
async def _generate_all() -> None:
    from agent import run

    OUTPUT_DIR.mkdir(exist_ok=True)
    for rfp_path in sample_rfps():
        print(f"\n=== generating for {rfp_path.name} ===")
        await run(str(rfp_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Run the agent on every sample RFP and write output/.")
    args = parser.parse_args()
    if args.generate:
        asyncio.run(_generate_all())
        return
    parser.print_help()


# ----------------------------------------------------------------------- assert
def _load_outputs() -> list[tuple[Path, dict]]:
    pairs = []
    for rfp_path in sample_rfps():
        out = output_for(rfp_path)
        if not out.exists():
            pytest.skip(f"output missing for {rfp_path.name}; run `python -m evals.eval --generate` first")
        pairs.append((rfp_path, json.loads(out.read_text())))
    return pairs


@pytest.fixture(scope="module")
def outputs() -> list[tuple[Path, dict]]:
    return _load_outputs()


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_output_exists(rfp_path: Path) -> None:
    assert output_for(rfp_path).exists(), \
        f"missing output for {rfp_path.name}; run `python -m evals.eval --generate`"


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_top_level_shape(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    rfp = json.loads(rfp_path.read_text())
    for key in ("rfp_id", "prospect", "generated_at", "answers"):
        assert key in out, f"{rfp_path.name}: missing key {key!r}"
    assert out["rfp_id"] == rfp["rfp_id"]
    assert isinstance(out["answers"], list) and out["answers"], "no answers produced"


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_answer_shape(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    required = {"question_id", "answer", "sources", "confidence", "needs_human_review"}
    for ans in out["answers"]:
        missing = required - ans.keys()
        assert not missing, f"answer {ans.get('question_id')} missing: {missing}"
        assert isinstance(ans["answer"], str) and ans["answer"].strip(), "empty answer"
        assert isinstance(ans["sources"], list)
        assert 0.0 <= float(ans["confidence"]) <= 1.0, "confidence out of [0,1]"
        assert isinstance(ans["needs_human_review"], bool)


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_low_confidence_flagged_for_review(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    for ans in out["answers"]:
        if float(ans["confidence"]) < 0.5:
            assert ans["needs_human_review"], (
                f"{ans['question_id']}: low confidence ({ans['confidence']}) "
                f"but needs_human_review is False"
            )


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_high_confidence_has_sources(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    for ans in out["answers"]:
        if float(ans["confidence"]) >= 0.7 and not ans["needs_human_review"]:
            assert ans["sources"], (
                f"{ans['question_id']}: high-confidence answer with no sources"
            )


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_human_review_has_reason(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    for ans in out["answers"]:
        if ans["needs_human_review"]:
            reason = (ans.get("review_reason") or "").strip()
            assert reason, f"{ans['question_id']}: flagged for review with no review_reason"


@pytest.mark.parametrize("rfp_path", sample_rfps(), ids=lambda p: p.stem)
def test_question_ids_match_input(rfp_path: Path) -> None:
    out = json.loads(output_for(rfp_path).read_text())
    rfp = json.loads(rfp_path.read_text())
    input_ids = {q["id"] for q in rfp["questions"]}
    output_ids = {a["question_id"] for a in out["answers"]}
    missing = input_ids - {oid.split(".")[0] for oid in output_ids}
    assert not missing, f"answers missing for input questions: {missing}"


if __name__ == "__main__":
    main()
