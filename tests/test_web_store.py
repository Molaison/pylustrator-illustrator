"""Real persistence, provenance, stale writes, and portable project history."""

import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from pylustrator.web.datasets import ValidationError, content_hash
from pylustrator.web.store import ConflictError, NotFoundError, ProjectStore

PDB = "ATOM      1  CA  ALA A   1      10.000  11.000  12.000  1.00 20.00           C  \nEND\n"


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path)


def test_reopen_preserves_source_derived_data_and_named_versions(store):
    project = store.create_project("研究项目", description="Notebook", tags=["paper", "paper"])
    pid = project["id"]
    project = store.add_dataset(pid, "原始数据.csv", "x,y\n1,2\n,4\n3,6\n")
    source = project["datasets"][0]
    version = store.save_version(pid, "Raw inputs", project["revision"])
    project = store.transform_dataset(pid, source["id"], [{"op": "drop_missing", "columns": ["x"]}], expected_hash=source["hash"])
    derived = project["datasets"][1]
    assert derived["sourceId"] == source["id"]
    assert derived["inputHash"] == source["hash"]
    assert derived["rowCount"] == 2
    reopened = ProjectStore(store.root)
    assert reopened.get_project(pid) == project
    assert reopened.get_dataset(pid, source["id"])["rowCount"] == 3
    assert reopened.list_versions(pid)[0]["id"] == version["id"]
    restored = reopened.restore_version(pid, version["id"], project["revision"])
    assert len(restored["datasets"]) == 1
    assert restored["revision"] > project["revision"]
    backup = reopened.list_versions(pid)[0]
    assert backup["automatic"] is True
    recovered = reopened.restore_version(pid, backup["id"], restored["revision"])
    assert len(recovered["datasets"]) == 2


def test_concurrent_saves_are_compare_and_swap(store):
    project = store.create_project("Concurrent")
    other = ProjectStore(store.root)
    def save(instance, name):
        try:
            return instance.update_project(project["id"], project["revision"], name=name)
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, instance, name) for instance, name in ((store, "A"), (other, "B"))]
        results = [f.result() for f in futures]
    assert sum(r is not None for r in results) == 1
    assert store.get_project(project["id"])["revision"] == 1


def test_failed_transform_and_stale_proposal_do_not_write(store):
    project = store.create_project("Immutable")
    project = store.add_dataset(project["id"], "Raw", "x\n1\n2\n")
    source = project["datasets"][0]
    with pytest.raises(ConflictError):
        store.transform_dataset(project["id"], source["id"], [], expected_hash="not-the-input")
    with pytest.raises(ValidationError):
        store.transform_dataset(project["id"], source["id"], [{"op": "eval"}])
    assert store.get_project(project["id"]) == project


def test_assets_are_deduplicated_and_project_scoped(store):
    first = store.create_project("First")
    second = store.create_project("Second")
    from PIL import Image
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(image, format="PNG")
    raw = image.getvalue()
    a = store.add_asset(first["id"], "Protein view", raw)
    b = store.add_asset(first["id"], "Second use", raw)
    assert a["asset"]["blobHash"] == b["asset"]["blobHash"]
    assert store.get_asset(first["id"], a["asset"]["id"])[0] == raw
    with pytest.raises(NotFoundError):
        store.get_asset(second["id"], a["asset"]["id"])
    with pytest.raises(ValidationError):
        store.add_asset(first["id"], "Unsafe", b"<svg/>", "image/svg+xml")


def test_bundle_roundtrip_carries_immutable_data_structures_and_history(store, tmp_path):
    project = store.create_project("Portable")
    pid = project["id"]
    project = store.add_dataset(pid, "Input", "g,x\na,1\nb,3\n")
    did = project["datasets"][0]["id"]
    project = store.transform_dataset(pid, did, [{"op": "derive", "column": "x", "target": "scaled", "method": "scale", "factor": 2}])
    project = store.add_structure(pid, "Chain", PDB, "pdb", metadata={"atomCount": 1, "chains": ["A"]}, view={"representation": "cartoon"})
    store.save_version(pid, "Complete", project["revision"])
    bundle = json.loads(json.dumps(store.export_bundle(pid)))
    imported_store = ProjectStore(tmp_path / "imported")
    imported = imported_store.import_bundle(bundle)
    assert imported["id"] != pid
    assert imported["document"] == project["document"]
    assert imported["datasets"] == project["datasets"]
    assert imported_store.dataset_rows(imported["id"], did) == store.dataset_rows(pid, did)
    sid = imported["structures"][0]["id"]
    assert imported_store.get_structure(imported["id"], sid)["content"] == PDB
    assert imported_store.list_versions(imported["id"])[0]["name"] == "Complete"


def test_tampered_bundles_are_rejected_atomically(store):
    project = store.create_project("Integrity")
    project = store.add_dataset(project["id"], "Raw", "x\n1\n2\n")
    bundle = store.export_bundle(project["id"])
    before = store.list_projects()
    tampered = copy.deepcopy(bundle)
    tampered["project"]["name"] = "Changed"
    with pytest.raises(ValidationError, match="checksum"):
        store.import_bundle(tampered)
    tampered = copy.deepcopy(bundle)
    digest = next(iter(tampered["blobs"]))
    tampered["blobs"][digest] = "dGFtcGVyZWQ="
    tampered["manifestHash"] = content_hash({k: v for k, v in tampered.items() if k != "manifestHash"})
    with pytest.raises(ValidationError, match="checksum"):
        store.import_bundle(tampered)
    assert store.list_projects() == before


def test_project_save_rejects_cross_project_plot_reference(store):
    first = store.create_project("First")
    first = store.add_dataset(first["id"], "Raw", "x,y\n1,2\n")
    second = store.create_project("Second")
    document = copy.deepcopy(second["document"])
    document["elements"] = [{"id": "plot-1", "type": "plot", "x": 0, "y": 0, "width": 300, "height": 200,
                             "plot": {"datasetId": first["datasets"][0]["id"], "kind": "line", "x": "x", "y": "y"}}]
    with pytest.raises(ValidationError, match="outside this project"):
        store.update_project(second["id"], second["revision"], document=document)
    assert store.get_project(second["id"]) == second


def test_structure_views_participate_in_revisions(store):
    project = store.create_project("Protein")
    project = store.add_structure(project["id"], "A", "ATOM test", "pdb", view={"representation": "cartoon"})
    sid = project["structures"][0]["id"]
    changed = store.update_structure(project["id"], sid, {"representation": "stick", "camera": [1, 2, 3]}, project["revision"])
    assert changed["revision"] == project["revision"] + 1
    with pytest.raises(ConflictError):
        store.update_structure(project["id"], sid, {}, project["revision"])


def test_bundle_replays_lineage_instead_of_trusting_declared_steps(store):
    project = store.create_project("Lineage")
    project = store.add_dataset(project["id"], "Raw", "x\n1\n2\n")
    project = store.transform_dataset(project["id"], project["datasets"][0]["id"], [
        {"op": "derive", "column": "x", "target": "y", "method": "scale", "factor": 2},
    ])
    bundle = store.export_bundle(project["id"])
    bundle["project"]["datasets"][1]["recipe"][0]["factor"] = 3
    bundle["manifestHash"] = content_hash({k: v for k, v in bundle.items() if k != "manifestHash"})
    with pytest.raises(ValidationError, match="cannot reproduce"):
        store.import_bundle(bundle)


def test_malformed_bundle_asset_reference_is_a_validation_error(store):
    project = store.create_project("Malformed")
    project = store.add_dataset(project["id"], "Raw", "x\n1\n")
    bundle = store.export_bundle(project["id"])
    bundle["project"]["datasets"][0]["blobHash"] = []
    bundle["manifestHash"] = content_hash({k: v for k, v in bundle.items() if k != "manifestHash"})
    with pytest.raises(ValidationError, match="references content"):
        store.import_bundle(bundle)


def test_invalid_png_and_invalid_bundled_structure_are_rejected(store):
    project = store.create_project("Invalid assets")
    with pytest.raises(ValidationError, match="valid PNG"):
        store.add_asset(project["id"], "Broken", b"\x89PNG\r\n\x1a\nnot-a-real-image")
    project = store.add_structure(project["id"], "Broken", "ATOM malformed", "pdb")
    bundle = store.export_bundle(project["id"])
    with pytest.raises(ValidationError, match="invalid molecular structure"):
        store.import_bundle(bundle)
