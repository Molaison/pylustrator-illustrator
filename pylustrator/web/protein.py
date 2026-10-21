"""Validated molecular source metadata and serializable viewer preferences."""

from __future__ import annotations

import math
import re
from typing import Any

MAX_STRUCTURE_BYTES = 16 * 1024 * 1024
MAX_ATOMS = 250_000


class StructureError(ValueError):
    """A molecular source or view cannot be used safely and meaningfully."""


def parse_structure(content: str, format: str = "pdb") -> dict[str, Any]:
    """Parse PDB/mmCIF using Gemmi, retaining the original source separately.

    Counts and sequences refer to the first model, matching the browser viewer.
    All model coordinates are checked; a multi-model input is never silently
    interpreted as a single enormous chain.
    """
    import gemmi

    if not isinstance(content, str) or not content.strip():
        raise StructureError("The structure file is empty.")
    if len(content.encode("utf-8")) > MAX_STRUCTURE_BYTES:
        raise StructureError("Structure files must be at most 16 MiB.")
    format = str(format).lower().lstrip(".")
    if format == "mmcif":
        format = "cif"
    if format not in {"pdb", "cif"}:
        raise StructureError("Choose a PDB or mmCIF structure file.")
    try:
        if format == "pdb":
            structure = gemmi.read_pdb_string(content)
        else:
            cif = gemmi.cif.read_string(content)
            if len(cif) != 1:
                raise StructureError("Import one mmCIF data block at a time.")
            structure = gemmi.make_structure_from_block(cif.sole_block())
    except StructureError:
        raise
    except (RuntimeError, ValueError, TypeError) as exc:
        raise StructureError(f"Could not parse {format.upper()} coordinates: {exc}") from exc
    if len(structure) == 0:
        raise StructureError("No molecular model was found in this file.")

    atom_total = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_total += 1
                    if atom_total > MAX_ATOMS:
                        raise StructureError("This workspace supports up to 250,000 atoms per source.")
                    xyz = (atom.pos.x, atom.pos.y, atom.pos.z)
                    if not all(math.isfinite(v) and abs(v) < 1e7 for v in xyz):
                        raise StructureError("Structure coordinates must be finite physical coordinates.")

    chains = []
    for chain in structure[0]:
        atoms = sum(len(residue) for residue in chain)
        if not atoms:
            continue
        sequence = ""
        protein_residues = 0
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info.is_amino_acid():
                sequence += info.one_letter_code.upper() or "X"
                protein_residues += 1
        chains.append({
            "id": chain.name,
            "name": chain.name or "(blank)",
            "atomCount": atoms,
            "residueCount": len(chain),
            "proteinResidueCount": protein_residues,
            "sequence": sequence,
        })
    if not chains:
        raise StructureError("No atoms with usable coordinates were found.")
    return {
        "format": format,
        "atomCount": sum(chain["atomCount"] for chain in chains),
        "residueCount": sum(chain["residueCount"] for chain in chains),
        "modelCount": len(structure),
        "chains": chains,
        "chainIds": [chain["id"] for chain in chains],
        "warnings": ["The viewer displays the first model."] if len(structure) > 1 else [],
    }


def validate_view(view: dict | None, chain_ids: list[str] | None = None) -> dict:
    """Return a bounded JSON-only 3Dmol view; never accept library internals."""
    if view is None:
        view = {}
    if not isinstance(view, dict):
        raise StructureError("A structure view must be an object.")
    result = {
        "representation": view.get("representation", "cartoon"),
        "color": view.get("color", "chain"),
        "chain": view.get("chain", "all"),
        "camera": view.get("camera"),
    }
    if result["representation"] not in {"cartoon", "stick", "sphere", "line"}:
        raise StructureError("Choose cartoon, stick, sphere, or line representation.")
    color = result["color"]
    if not isinstance(color, str) or (
        color not in {"chain", "spectrum", "element", "secondary", "bfactor"}
        and not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
    ):
        raise StructureError("Choose a supported molecular color scheme.")
    chain = result["chain"]
    if not isinstance(chain, str) or len(chain) > 64:
        raise StructureError("A chain identifier must be a short string.")
    if chain_ids is not None and chain != "all" and chain not in chain_ids:
        raise StructureError("The selected chain is not present in this structure.")
    camera = result["camera"]
    if camera is not None:
        if not isinstance(camera, list) or len(camera) != 8 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 1e8 for value in camera
        ):
            raise StructureError("A saved camera must contain eight finite numbers.")
        result["camera"] = [float(value) for value in camera]
    return result
