from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.artist import Artist

from pylustrator.interaction import (
    HitCandidate,
    HitStack,
    SelectionKernel,
    SelectionMode,
    TopHitStatus,
)


@dataclass
class _Scene:
    artists: dict[str, Artist]
    kernel: SelectionKernel


class _CountingHits(Iterable[HitCandidate]):
    def __init__(self, candidates: Iterable[HitCandidate]) -> None:
        self._candidates = tuple(candidates)
        self.consumed = 0

    def __iter__(self) -> Iterator[HitCandidate]:
        for candidate in self._candidates:
            self.consumed += 1
            yield candidate


def _scene(
    *,
    names: Iterable[str],
    groups: Iterable[str] = (),
    parents: Iterable[tuple[str, str]] = (),
) -> _Scene:
    artists = {name: Artist() for name in names}
    group_ids = {id(artists[name]) for name in groups}
    parent_by_id = {
        id(artists[child]): artists[parent] for child, parent in parents
    }
    kernel = SelectionKernel(
        parent_of=lambda artist: parent_by_id.get(id(artist)),
        is_group=lambda artist: id(artist) in group_ids,
        label_of=lambda artist: next(
            name for name, candidate in artists.items() if candidate is artist
        ),
    )
    return _Scene(artists, kernel)


def _stack(
    scene: _Scene,
    hits: Iterable[tuple[str, bool]],
) -> HitStack:
    return HitStack(
        tuple(
            HitCandidate(
                artist=scene.artists[name],
                editable=editable,
                draw_key=(index,),
                registration_index=index,
            )
            for index, (name, editable) in enumerate(hits)
        )
    )


@pytest.mark.parametrize(
    (
        "mode",
        "scope",
        "groups",
        "parents",
        "hits",
        "expected_target",
        "expected_raw_leaf",
        "expected_blocked",
    ),
    [
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (("leaf", True),),
            "leaf",
            "leaf",
            False,
            id="object-leaf",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            None,
            (),
            (),
            (("leaf", True),),
            "leaf",
            "leaf",
            False,
            id="direct-leaf",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            None,
            ("group",),
            (("leaf", "group"),),
            (("leaf", True), ("group", True)),
            "group",
            "leaf",
            False,
            id="object-promotes-leaf-to-group",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            None,
            ("group",),
            (("leaf", "group"),),
            (("leaf", True), ("group", True)),
            "leaf",
            "leaf",
            False,
            id="direct-keeps-exact-leaf",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            "scope",
            ("scope",),
            (("inside", "scope"),),
            (("outside", True), ("scope", True), ("inside", True)),
            "inside",
            "inside",
            False,
            id="object-ignores-outside-and-scope-root",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            "scope",
            ("scope",),
            (("inside", "scope"),),
            (("outside", True), ("scope", True), ("inside", True)),
            "inside",
            "inside",
            False,
            id="direct-ignores-outside-and-scope-root",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (("barrier", False), ("behind", True)),
            None,
            None,
            True,
            id="object-unsupported-barrier",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            "scope",
            ("scope",),
            (("inside", "scope"),),
            (("outside", False), ("scope", False), ("inside", False)),
            None,
            None,
            True,
            id="direct-scope-local-unsupported-barrier",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (),
            None,
            None,
            False,
            id="empty-stream",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            "scope",
            ("scope",),
            (),
            (("outside", True), ("scope", True)),
            None,
            None,
            False,
            id="no-relevant-hit",
        ),
    ],
)
def test_resolve_top_matches_full_resolver_for_conclusive_scenarios(
    mode: SelectionMode,
    scope: str | None,
    groups: tuple[str, ...],
    parents: tuple[tuple[str, str], ...],
    hits: tuple[tuple[str, bool], ...],
    expected_target: str | None,
    expected_raw_leaf: str | None,
    expected_blocked: bool,
) -> None:
    names = {
        *(name for name, _editable in hits),
        *groups,
        *(child for child, _parent in parents),
        *(parent for _child, parent in parents),
    }
    scene = _scene(names=names, groups=groups, parents=parents)
    scene.kernel.set_mode(mode)
    if scope is not None:
        assert scene.kernel.enter_isolation(scene.artists[scope])
    hit_stack = _stack(scene, hits)

    full = scene.kernel.resolve(hit_stack)
    top = scene.kernel.resolve_top(iter(hit_stack))

    assert top.status is TopHitStatus.RESOLVED
    assert top.target is full.target
    assert top.raw_leaf is full.raw_leaf
    assert top.blocked is full.blocked
    assert top.target is (
        scene.artists[expected_target] if expected_target is not None else None
    )
    assert top.raw_leaf is (
        scene.artists[expected_raw_leaf]
        if expected_raw_leaf is not None
        else None
    )
    assert top.blocked is expected_blocked


@pytest.mark.parametrize("has_descendant_leaf", [True, False])
def test_direct_group_shell_requests_full_stack_fallback(
    has_descendant_leaf: bool,
) -> None:
    scene = _scene(
        names=("group", "leaf"),
        groups=("group",),
        parents=(("leaf", "group"),),
    )
    scene.kernel.set_mode(SelectionMode.DIRECT)
    hits = [("group", True)]
    if has_descendant_leaf:
        hits.append(("leaf", True))
    hit_stack = _stack(scene, hits)

    full = scene.kernel.resolve(hit_stack)
    top = scene.kernel.resolve_top(iter(hit_stack))

    assert top.status is TopHitStatus.NEEDS_FULL_STACK
    assert top.target is None
    assert top.raw_leaf is scene.artists["group"]
    assert not top.blocked
    if has_descendant_leaf:
        assert full.target is scene.artists["leaf"]
        assert not full.blocked
    else:
        assert full.target is None
        assert full.blocked


@pytest.mark.parametrize(
    ("mode", "scope", "groups", "parents", "hits", "expected_consumed"),
    [
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (("front", True), ("behind", True)),
            1,
            id="object-first-hit",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            None,
            (),
            (),
            (("front", True), ("behind", True)),
            1,
            id="direct-leaf",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (("barrier", False), ("behind", True)),
            1,
            id="unsupported-barrier",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            "scope",
            ("scope",),
            (("inside", "scope"),),
            (
                ("outside", True),
                ("scope", True),
                ("inside", True),
                ("behind", True),
            ),
            3,
            id="out-of-scope-and-root-before-decision",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            None,
            ("group",),
            (("leaf", "group"),),
            (("group", True), ("leaf", True)),
            1,
            id="direct-group-shell-fallback",
        ),
        pytest.param(
            SelectionMode.OBJECT,
            None,
            (),
            (),
            (),
            0,
            id="empty-stream",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            "scope",
            ("scope",),
            (),
            (("outside-a", True), ("outside-b", False), ("scope", True)),
            3,
            id="all-hits-irrelevant",
        ),
    ],
)
def test_resolve_top_consumes_only_through_first_decisive_hit(
    mode: SelectionMode,
    scope: str | None,
    groups: tuple[str, ...],
    parents: tuple[tuple[str, str], ...],
    hits: tuple[tuple[str, bool], ...],
    expected_consumed: int,
) -> None:
    names = {
        *(name for name, _editable in hits),
        *groups,
        *(child for child, _parent in parents),
        *(parent for _child, parent in parents),
    }
    scene = _scene(names=names, groups=groups, parents=parents)
    scene.kernel.set_mode(mode)
    if scope is not None:
        assert scene.kernel.enter_isolation(scene.artists[scope])
    counting_hits = _CountingHits(_stack(scene, hits))

    scene.kernel.resolve_top(counting_hits)

    assert counting_hits.consumed == expected_consumed


@pytest.mark.parametrize(
    ("mode", "expected_candidates"),
    [
        pytest.param(
            SelectionMode.OBJECT,
            ("group", "loose"),
            id="object-order-and-group-deduplication",
        ),
        pytest.param(
            SelectionMode.DIRECT,
            ("leaf-a", "leaf-b", "loose"),
            id="direct-leaf-order",
        ),
    ],
)
def test_full_hit_stack_resolution_order_and_behavior_remain_unchanged(
    mode: SelectionMode,
    expected_candidates: tuple[str, ...],
) -> None:
    scene = _scene(
        names=("leaf-a", "leaf-b", "group", "loose", "barrier", "behind"),
        groups=("group",),
        parents=(("leaf-a", "group"), ("leaf-b", "group")),
    )
    scene.kernel.set_mode(mode)
    hit_stack = _stack(
        scene,
        (
            ("leaf-a", True),
            ("leaf-b", True),
            ("loose", True),
            ("barrier", False),
            ("behind", True),
        ),
    )

    resolution = scene.kernel.resolve(hit_stack)

    assert len(hit_stack) == 5
    assert hit_stack.artists == tuple(
        scene.artists[name]
        for name in ("leaf-a", "leaf-b", "loose", "barrier", "behind")
    )
    assert tuple(hit_stack) == hit_stack.candidates
    assert resolution.hit_stack is hit_stack
    assert resolution.raw_leaf is scene.artists["leaf-a"]
    assert resolution.candidates == tuple(
        scene.artists[name] for name in expected_candidates
    )
    assert resolution.target is scene.artists[expected_candidates[0]]
    assert not resolution.blocked
