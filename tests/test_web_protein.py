from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from pylustrator.web.protein import StructureError, parse_structure, validate_view  # noqa: E402


SAMPLE = Path(__file__).parents[1] / "pylustrator/web/examples/1crn.pdb"


def test_real_pdb_and_mmcif_metadata_agree():
    source = SAMPLE.read_text()
    pdb = parse_structure(source, "pdb")
    cif = parse_structure(gemmi.read_pdb_string(source).make_mmcif_document().as_string(), "cif")
    assert pdb["atomCount"] == cif["atomCount"] == 327
    assert pdb["chains"] == cif["chains"]
    assert pdb["residueCount"] == 46
    assert len(pdb["chains"][0]["sequence"]) == 46


@pytest.mark.parametrize("source,format", [("", "pdb"), ("not a structure", "pdb"), ("data_broken", "cif"), ("{}", "json")])
def test_reject_empty_or_invalid_structure(source, format):
    with pytest.raises(StructureError):
        parse_structure(source, format)


def test_view_is_bounded_and_chain_is_validated():
    assert validate_view({}, ["A"])["representation"] == "cartoon"
    view = {"representation": "stick", "chain": "A", "camera": [0, 0, 0, -50, 0, 0, 0, 1]}
    assert validate_view(view, ["A"])["camera"][-1] == 1
    for invalid in [{"chain": "B"}, {"camera": [float("nan")] * 8}, {"camera": [0] * 100}, {"color": "url(javascript:alert(1))"}]:
        with pytest.raises(StructureError):
            validate_view(invalid, ["A"])
