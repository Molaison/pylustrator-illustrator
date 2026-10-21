"""Transactional research projects and immutable content-addressed data.

SQLite owns both metadata and blobs so a checkpoint cannot outlive its source
files. Each write reads the latest project inside an IMMEDIATE transaction.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import (
    ValidationError, apply_recipe, canonical_json, content_hash, parse_dataset,
    preview_table, table_from_rows,
)

MAX_BLOB_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_PROJECT_RECORDS = 1000


class NotFoundError(LookupError):
    """The requested object does not belong to this project."""


class ConflictError(RuntimeError):
    """An optimistic save or AI proposal targets an obsolete revision."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _name(value: Any, label: str = "Name") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValidationError(f"{label} must contain 1–512 characters.")
    return value.strip()


def _json(value: Any, limit: int = 2 * 1024 * 1024) -> str:
    result = canonical_json(value)
    if len(result.encode("utf-8")) > limit:
        raise ValidationError("Metadata exceeds the permitted size.")
    return result


def _revision(expected: Any, current: int) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ValidationError("A nonnegative integer revision is required.")
    if expected != current:
        raise ConflictError(f"Project has revision {current}; reload before saving revision {expected}.")


def _document(document: Any) -> dict:
    from .document import validate_document
    try:
        return validate_document(document)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _new_document() -> dict:
    from .document import new_document
    return new_document()


def _tags(tags: Any) -> list[str]:
    if not isinstance(tags, list) or len(tags) > 50 or any(not isinstance(t, str) or len(t) > 128 for t in tags):
        raise ValidationError("Tags must be at most 50 short strings.")
    return list(dict.fromkeys(t.strip() for t in tags if t.strip()))


def _description(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 50_000:
        raise ValidationError("Description must be text of at most 50,000 characters.")
    return value


def _validate_png(data: bytes) -> None:
    from PIL import Image, UnidentifiedImageError
    if not isinstance(data, bytes) or len(data) > MAX_BLOB_BYTES:
        raise ValidationError("A PNG asset may contain at most 32 MiB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > 32_000_000:
                raise ValidationError("Upload a PNG image of at most 32 megapixels.")
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValidationError("The image asset is not a valid PNG.") from exc


class ProjectStore:
    """A durable local workspace. Never use a client path as a storage path."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "workspace.sqlite3"
        self._lock = threading.RLock()
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    revision INTEGER NOT NULL, updated_at TEXT NOT NULL,
                    body TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY, data BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                    name TEXT NOT NULL, created_at TEXT NOT NULL,
                    revision INTEGER NOT NULL, body TEXT NOT NULL, summary TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS versions_by_project ON versions(project_id, created_at);
            """)
            if "summary" not in {r["name"] for r in conn.execute("PRAGMA table_info(versions)")}:
                conn.execute("ALTER TABLE versions ADD COLUMN summary TEXT")
            for row in conn.execute("SELECT id,body FROM versions WHERE summary IS NULL").fetchall():
                conn.execute("UPDATE versions SET summary=? WHERE id=?",
                             (canonical_json(json.loads(row["body"])["version"]), row["id"]))

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _read(conn, project_id: str) -> dict:
        row = conn.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise NotFoundError("Project not found.")
        return json.loads(row["body"])

    @staticmethod
    def _put_blob(conn, data: bytes) -> str:
        if not isinstance(data, bytes) or len(data) > MAX_BLOB_BYTES:
            raise ValidationError("An asset may contain at most 32 MiB.")
        digest = hashlib.sha256(data).hexdigest()
        conn.execute("INSERT OR IGNORE INTO blobs(hash,data) VALUES (?,?)", (digest, data))
        return digest

    @staticmethod
    def _blob(conn, digest: str) -> bytes:
        row = conn.execute("SELECT data FROM blobs WHERE hash=?", (digest,)).fetchone()
        if row is None:
            raise ValidationError("Project references missing content.")
        return bytes(row["data"])

    @staticmethod
    def _member(project: dict, category: str, item_id: str) -> dict:
        for item in project[category]:
            if item["id"] == item_id:
                return item
        raise NotFoundError(f"{category.rstrip('s').capitalize()} not found in this project.")

    @staticmethod
    def _write(conn, project: dict, *, advance: bool = True) -> dict:
        if advance:
            project["revision"] += 1
            project["updatedAt"] = _now()
        conn.execute(
            "INSERT INTO projects(id,name,revision,updated_at,body) VALUES(?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name,revision=excluded.revision,"
            "updated_at=excluded.updated_at,body=excluded.body",
            (project["id"], project["name"], project["revision"], project["updatedAt"], _json(project, 8 * 1024 * 1024)),
        )
        return deepcopy(project)

    @staticmethod
    def _check_references(project: dict) -> None:
        dataset_ids = {d["id"] for d in project["datasets"]}
        asset_ids = {a["id"] for a in project["assets"]}
        for element in project["document"].get("elements", []):
            if element.get("type") == "plot" and element.get("plot", {}).get("datasetId") not in dataset_ids:
                raise ValidationError("A figure plot references a dataset outside this project.")
            if element.get("type") == "image" and element.get("assetId") not in asset_ids:
                raise ValidationError("A figure image references an asset outside this project.")

    def list_projects(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id,name,revision,updated_at FROM projects ORDER BY updated_at DESC,id").fetchall()
        return [{"id": r["id"], "name": r["name"], "revision": r["revision"], "updatedAt": r["updated_at"]} for r in rows]

    def create_project(self, name: str, description: str = "", document: dict | None = None,
                       tags: list[str] | None = None) -> dict:
        timestamp = _now()
        project = {"id": _id("project"), "name": _name(name), "description": _description(description),
                   "tags": _tags(tags or []), "revision": 0, "createdAt": timestamp, "updatedAt": timestamp,
                   "document": _document(document) if document is not None else _new_document(),
                   "datasets": [], "structures": [], "assets": []}
        self._check_references(project)
        with self._transaction() as conn:
            return self._write(conn, project, advance=False)

    def get_project(self, project_id: str) -> dict:
        with closing(self._connect()) as conn:
            return self._read(conn, project_id)

    def update_project(self, project_id: str, expected_revision: int, **changes) -> dict:
        if set(changes) - {"name", "description", "tags", "document"}:
            raise ValidationError("Only project name, description, tags, and document can be changed.")
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            _revision(expected_revision, project["revision"])
            if "name" in changes:
                project["name"] = _name(changes["name"])
            if "description" in changes:
                project["description"] = _description(changes["description"])
            if "tags" in changes:
                project["tags"] = _tags(changes["tags"])
            if "document" in changes:
                project["document"] = _document(changes["document"])
            self._check_references(project)
            return self._write(conn, project)

    def _add_table(self, conn, project: dict, name: str, table: dict, *, raw: bytes | None = None,
                   source: dict | None = None, format: str = "json", provenance: dict | None = None) -> dict:
        if len(project["datasets"]) >= MAX_PROJECT_RECORDS:
            raise ValidationError("Project dataset limit reached.")
        metadata = {"id": _id("data"), "name": _name(name), "format": format, "createdAt": _now(),
                    "hash": table["hash"], "columns": table["columns"], "rowCount": table["rowCount"],
                    "blobHash": self._put_blob(conn, canonical_json(table).encode("utf-8"))}
        if raw is not None:
            metadata["sourceHash"] = self._put_blob(conn, raw)
        if source is not None:
            metadata.update(sourceId=source["id"], inputHash=source["hash"], recipe=deepcopy(table.get("recipe", [])))
        if provenance is not None:
            _json(provenance, 128 * 1024)
            metadata["provenance"] = deepcopy(provenance)
        project["datasets"].append(metadata)
        return metadata

    def add_dataset(self, project_id: str, name: str, content: str | bytes, format: str = "csv",
                    expected_revision: int | None = None) -> dict:
        table = parse_dataset(content, format)
        raw = content.encode("utf-8") if isinstance(content, str) else content
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            if expected_revision is not None:
                _revision(expected_revision, project["revision"])
            self._add_table(conn, project, name, table, raw=raw, format=format)
            return self._write(conn, project)

    def _table(self, conn, project: dict, dataset_id: str) -> dict:
        metadata = self._member(project, "datasets", dataset_id)
        table = json.loads(self._blob(conn, metadata["blobHash"]))
        return {**table, **deepcopy(metadata)}

    def get_dataset_data(self, project_id: str, dataset_id: str) -> dict:
        with closing(self._connect()) as conn:
            project = self._read(conn, project_id)
            return self._table(conn, project, dataset_id)

    def dataset_rows(self, project_id: str, dataset_id: str) -> list[dict]:
        return self.get_dataset_data(project_id, dataset_id)["rows"]

    def get_dataset(self, project_id: str, dataset_id: str, offset: int = 0, limit: int = 100) -> dict:
        return preview_table(self.get_dataset_data(project_id, dataset_id), offset, limit)

    def preview_transform(self, project_id: str, dataset_id: str, steps: list[dict],
                          offset: int = 0, limit: int = 100) -> dict:
        source = self.get_dataset_data(project_id, dataset_id)
        table = apply_recipe(source, steps)
        table.update(name=f"{source['name']} · preview", sourceId=dataset_id)
        return preview_table(table, offset, limit)

    def transform_dataset(self, project_id: str, dataset_id: str, steps: list[dict], name: str | None = None,
                          expected_hash: str | None = None, expected_revision: int | None = None,
                          provenance: dict | None = None) -> dict:
        # Computation is outside the write transaction. Sources are immutable.
        source = self.get_dataset_data(project_id, dataset_id)
        if expected_hash is not None and source["hash"] != expected_hash:
            raise ConflictError("The proposal targets a different dataset version. Request a new preview.")
        table = apply_recipe(source, steps)
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            if expected_revision is not None:
                _revision(expected_revision, project["revision"])
            current_source = self._member(project, "datasets", dataset_id)
            if current_source["hash"] != source["hash"]:
                raise ConflictError("Dataset changed while processing.")
            self._add_table(conn, project, name or f"{source['name']} · processed", table,
                            source=source, provenance=provenance)
            return self._write(conn, project)

    def add_structure(self, project_id: str, name: str, content: str, format: str,
                      metadata: dict | None = None, view: dict | None = None,
                      expected_revision: int | None = None) -> dict:
        if not isinstance(format, str) or format not in {"pdb", "cif"} or not isinstance(content, str) or not content.strip():
            raise ValidationError("A nonempty PDB or mmCIF structure is required.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError("Structure metadata must be an object.")
        if view is not None and not isinstance(view, dict):
            raise ValidationError("Structure view must be an object.")
        _json(metadata or {}, 1024 * 1024)
        _json(view or {}, 128 * 1024)
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            if expected_revision is not None:
                _revision(expected_revision, project["revision"])
            if len(project["structures"]) >= MAX_PROJECT_RECORDS:
                raise ValidationError("Project structure limit reached.")
            digest = self._put_blob(conn, content.encode("utf-8"))
            item = {**deepcopy(metadata or {}), "id": _id("structure"), "name": _name(name),
                    "format": format, "hash": digest, "blobHash": digest,
                    "createdAt": _now(), "view": deepcopy(view or {})}
            item.pop("content", None)
            project["structures"].append(item)
            return self._write(conn, project)

    def get_structure(self, project_id: str, structure_id: str) -> dict:
        with closing(self._connect()) as conn:
            project = self._read(conn, project_id)
            item = self._member(project, "structures", structure_id)
            return {**deepcopy(item), "content": self._blob(conn, item["blobHash"]).decode("utf-8")}

    def update_structure(self, project_id: str, structure_id: str, view: dict,
                         expected_revision: int) -> dict:
        if not isinstance(view, dict):
            raise ValidationError("Structure view must be an object.")
        _json(view, 128 * 1024)
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            _revision(expected_revision, project["revision"])
            item = self._member(project, "structures", structure_id)
            item["view"] = deepcopy(view)
            return self._write(conn, project)

    def add_asset(self, project_id: str, name: str, data: bytes, mime_type: str = "image/png",
                  provenance: dict | None = None, expected_revision: int | None = None) -> dict:
        if mime_type != "image/png" or not isinstance(data, bytes) or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValidationError("Only PNG image assets are supported.")
        _validate_png(data)
        _json(provenance or {}, 128 * 1024)
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            if expected_revision is not None:
                _revision(expected_revision, project["revision"])
            if len(project["assets"]) >= MAX_PROJECT_RECORDS:
                raise ValidationError("Project asset limit reached.")
            digest = self._put_blob(conn, data)
            asset = {"id": _id("asset"), "name": _name(name), "mimeType": mime_type, "hash": digest,
                     "blobHash": digest, "size": len(data), "createdAt": _now(),
                     "provenance": deepcopy(provenance or {})}
            project["assets"].append(asset)
            return {"asset": deepcopy(asset), "project": self._write(conn, project)}

    def get_asset(self, project_id: str, asset_id: str) -> tuple[bytes, dict]:
        with closing(self._connect()) as conn:
            project = self._read(conn, project_id)
            asset = self._member(project, "assets", asset_id)
            return self._blob(conn, asset["blobHash"]), deepcopy(asset)

    @staticmethod
    def _version(conn, project: dict, name: str, *, automatic: bool = False) -> dict:
        version = {"id": _id("version"), "name": _name(name, "Checkpoint name"),
                   "createdAt": _now(), "revision": project["revision"], "automatic": automatic,
                   "datasetCount": len(project["datasets"]), "structureCount": len(project["structures"]),
                   "elementCount": len(project["document"].get("elements", [])), "hash": content_hash(project)}
        conn.execute("INSERT INTO versions(id,project_id,name,created_at,revision,body,summary) VALUES(?,?,?,?,?,?,?)",
                     (version["id"], project["id"], version["name"], version["createdAt"],
                      version["revision"], _json({"version": version, "snapshot": project}, 10 * 1024 * 1024),
                      canonical_json(version)))
        return version

    def save_version(self, project_id: str, name: str, expected_revision: int) -> dict:
        with self._transaction() as conn:
            project = self._read(conn, project_id)
            _revision(expected_revision, project["revision"])
            return self._version(conn, project, name)

    def list_versions(self, project_id: str) -> list[dict]:
        with closing(self._connect()) as conn:
            self._read(conn, project_id)
            rows = conn.execute("SELECT summary FROM versions WHERE project_id=? ORDER BY created_at DESC,rowid DESC",
                                (project_id,)).fetchall()
        return [json.loads(r["summary"]) for r in rows]

    def get_version(self, project_id: str, version_id: str) -> dict:
        with closing(self._connect()) as conn:
            self._read(conn, project_id)
            row = conn.execute("SELECT body FROM versions WHERE project_id=? AND id=?", (project_id, version_id)).fetchone()
        if row is None:
            raise NotFoundError("Checkpoint not found in this project.")
        return json.loads(row["body"])

    def restore_version(self, project_id: str, version_id: str, expected_revision: int) -> dict:
        with self._transaction() as conn:
            current = self._read(conn, project_id)
            _revision(expected_revision, current["revision"])
            row = conn.execute("SELECT body FROM versions WHERE project_id=? AND id=?", (project_id, version_id)).fetchone()
            if row is None:
                raise NotFoundError("Checkpoint not found in this project.")
            record = json.loads(row["body"])
            restored = record["snapshot"]
            self._version(conn, current, f"Before restoring {record['version']['name']}"[:512], automatic=True)
            restored["revision"] = current["revision"]
            restored["id"] = project_id
            return self._write(conn, restored)

    @staticmethod
    def _hashes(project: dict) -> set[str]:
        result = set()
        for category in ("datasets", "structures", "assets"):
            for item in project[category]:
                result.add(item["blobHash"])
                if "sourceHash" in item:
                    result.add(item["sourceHash"])
        return result

    def export_bundle(self, project_id: str) -> dict:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN")
            project = self._read(conn, project_id)
            records = [json.loads(r["body"]) for r in conn.execute(
                "SELECT body FROM versions WHERE project_id=? ORDER BY created_at,rowid", (project_id,)).fetchall()]
            hashes = self._hashes(project)
            for record in records:
                hashes.update(self._hashes(record["snapshot"]))
            blobs = {}
            total = 0
            for digest in sorted(hashes):
                data = self._blob(conn, digest)
                total += len(data)
                if total > MAX_BUNDLE_BYTES:
                    raise ValidationError("This project exceeds the 64 MiB portable bundle limit.")
                blobs[digest] = base64.b64encode(data).decode("ascii")
            conn.commit()
        bundle = {"format": "pylustrator-project", "schemaVersion": 1,
                  "project": project, "versions": records, "blobs": blobs}
        bundle["manifestHash"] = content_hash(bundle)
        return bundle

    def _validate_snapshot(self, snapshot: Any, blobs: dict[str, bytes], cache: dict) -> dict:
        if not isinstance(snapshot, dict):
            raise ValidationError("Bundle project metadata is invalid.")
        result = {"id": _name(snapshot.get("id")), "name": _name(snapshot.get("name")),
                  "description": _description(snapshot.get("description", "")),
                  "tags": _tags(snapshot.get("tags", [])), "revision": snapshot.get("revision"),
                  "createdAt": snapshot.get("createdAt"), "updatedAt": snapshot.get("updatedAt"),
                  "document": _document(snapshot.get("document"))}
        _revision(result["revision"], result["revision"])
        if any(not isinstance(result[key], str) or len(result[key]) > 80 for key in ("createdAt", "updatedAt")):
            raise ValidationError("Bundle timestamps are invalid.")
        for category in ("datasets", "structures", "assets"):
            items = snapshot.get(category)
            if not isinstance(items, list) or len(items) > MAX_PROJECT_RECORDS:
                raise ValidationError("Bundle asset collection is invalid.")
            result[category] = deepcopy(items)
            ids = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValidationError("Bundle asset metadata is invalid.")
                item_id = _name(item.get("id"))
                _name(item.get("name"))
                if item_id in ids:
                    raise ValidationError("Bundle contains duplicate asset IDs.")
                ids.add(item_id)
                if (not isinstance(item.get("blobHash"), str) or item["blobHash"] not in blobs
                        or ("sourceHash" in item and (not isinstance(item["sourceHash"], str) or item["sourceHash"] not in blobs))):
                    raise ValidationError("Bundle references content that was not included.")
                if category == "datasets":
                    try:
                        if item["blobHash"] not in cache["tables"]:
                            data = json.loads(blobs[item["blobHash"]])
                            columns = [c["name"] for c in data["columns"]]
                            validated = table_from_rows(data["rows"], columns)
                            if data.get("hash") != validated["hash"]:
                                raise ValidationError("Dataset content disagrees with its hash.")
                            cache["tables"][item["blobHash"]] = validated
                        validated = cache["tables"][item["blobHash"]]
                    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValidationError("Bundle contains invalid dataset content.") from exc
                    if (item.get("hash") != validated["hash"]
                            or item.get("columns") != validated["columns"] or item.get("rowCount") != validated["rowCount"]):
                        raise ValidationError("Dataset content disagrees with its metadata.")
                elif item.get("hash") != item["blobHash"]:
                    raise ValidationError("Asset content disagrees with its metadata.")
                elif category == "assets":
                    if item.get("mimeType") != "image/png":
                        raise ValidationError("Bundle image is not a PNG asset.")
                    if item["blobHash"] not in cache["images"]:
                        _validate_png(blobs[item["blobHash"]])
                        cache["images"].add(item["blobHash"])
                else:
                    from .protein import parse_structure, validate_view
                    if not isinstance(item.get("format"), str) or item["format"] not in {"pdb", "cif"}:
                        raise ValidationError("Bundle structure has an invalid format.")
                    try:
                        key = (item["blobHash"], item["format"])
                        if key not in cache["structures"]:
                            cache["structures"][key] = parse_structure(blobs[item["blobHash"]].decode("utf-8"), item["format"])
                        structure = cache["structures"][key]
                        validate_view(item.get("view"), structure["chainIds"])
                        if "atomCount" in item and item["atomCount"] != structure["atomCount"]:
                            raise ValidationError("Structure metadata disagrees with its coordinates.")
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise ValidationError(f"Bundle contains an invalid molecular structure: {exc}") from exc
            if category == "datasets":
                tables = {item["id"]: cache["tables"][item["blobHash"]] for item in items}
                by_id = {item["id"]: item for item in items}
                checked, checking = set(), set()
                def verify_lineage(item_id):
                    if item_id in checked:
                        return
                    if item_id in checking:
                        raise ValidationError("Dataset lineage contains a cycle.")
                    if len(checking) >= 200:
                        raise ValidationError("Dataset lineage exceeds the 200-generation import limit.")
                    checking.add(item_id)
                    item = by_id[item_id]
                    if "sourceId" in item:
                        if item["sourceId"] not in by_id:
                            raise ValidationError("Derived dataset has no source in this project snapshot.")
                        verify_lineage(item["sourceId"])
                        source = by_id[item["sourceId"]]
                        if item.get("inputHash") != source["hash"]:
                            raise ValidationError("Dataset lineage refers to the wrong source hash.")
                        replay_key = content_hash({"input": source["hash"], "recipe": item.get("recipe")})
                        if replay_key not in cache["recipes"]:
                            cache["recipes"][replay_key] = apply_recipe(tables[item["sourceId"]], item.get("recipe"))["hash"]
                        if cache["recipes"][replay_key] != item["hash"]:
                            raise ValidationError("Dataset recipe cannot reproduce its saved content.")
                    elif "sourceHash" in item:
                        raw_key = (item["sourceHash"], item.get("format"))
                        if not isinstance(raw_key[1], str):
                            raise ValidationError("Raw dataset format is invalid.")
                        if raw_key not in cache["raw"]:
                            cache["raw"][raw_key] = parse_dataset(blobs[item["sourceHash"]], item["format"])["hash"]
                        if cache["raw"][raw_key] != item["hash"]:
                            raise ValidationError("Dataset source cannot reproduce its saved content.")
                    else:
                        raise ValidationError("Dataset is missing source provenance.")
                    checking.remove(item_id)
                    checked.add(item_id)
                for item in items:
                    verify_lineage(item["id"])
        self._check_references(result)
        return result

    def import_bundle(self, bundle: Any) -> dict:
        if (not isinstance(bundle, dict) or bundle.get("format") != "pylustrator-project"
                or type(bundle.get("schemaVersion")) is not int or bundle["schemaVersion"] != 1):
            raise ValidationError("Unsupported project bundle format.")
        unsigned = {k: v for k, v in bundle.items() if k != "manifestHash"}
        if bundle.get("manifestHash") != content_hash(unsigned):
            raise ValidationError("Project bundle checksum does not match its contents.")
        encoded = bundle.get("blobs")
        if not isinstance(encoded, dict) or len(encoded) > 10_000:
            raise ValidationError("Bundle content table is invalid.")
        blobs = {}
        total = 0
        for digest, value in encoded.items():
            if not isinstance(digest, str) or len(digest) != 64 or not isinstance(value, str) or len(value) > MAX_BLOB_BYTES * 4 // 3 + 4:
                raise ValidationError("Bundle blob exceeds the permitted size or has an invalid hash.")
            try:
                raw = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValidationError("Bundle contains invalid base64 content.") from exc
            total += len(raw)
            if len(raw) > MAX_BLOB_BYTES or total > MAX_BUNDLE_BYTES:
                raise ValidationError("Bundle exceeds the 64 MiB import limit.")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValidationError("Bundle content checksum mismatch.")
            blobs[digest] = raw
        cache = {"tables": {}, "images": set(), "structures": {}, "recipes": {}, "raw": {}}
        project = self._validate_snapshot(bundle.get("project"), blobs, cache)
        versions = bundle.get("versions", [])
        if not isinstance(versions, list) or len(versions) > 1000:
            raise ValidationError("Bundle may contain at most 1000 checkpoints.")
        snapshots = []
        for record in versions:
            if not isinstance(record, dict) or not isinstance(record.get("version"), dict):
                raise ValidationError("Bundle checkpoint is invalid.")
            snapshot = self._validate_snapshot(record.get("snapshot"), blobs, cache)
            if record["version"].get("hash") != content_hash(snapshot):
                raise ValidationError("Checkpoint checksum mismatch.")
            snapshots.append((_name(record["version"].get("name")), snapshot))
        project["id"] = _id("project")
        project["name"] = _name(f"{project['name']} (imported)"[:512])
        project["revision"] = 0
        project["updatedAt"] = _now()
        with self._transaction() as conn:
            for data in blobs.values():
                self._put_blob(conn, data)
            self._write(conn, project, advance=False)
            for name, snapshot in snapshots:
                snapshot["id"] = project["id"]
                self._version(conn, snapshot, name)
        return deepcopy(project)
