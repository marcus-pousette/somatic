"""Unit tests for the tree-speculation plumbing that doesn't need MLX:
control-frame codec, ancestor masks, path acceptance, draft-cache keep logic."""

import pytest

from soup.serving.mlx_engine import tree_accept, tree_draft_keep
from soup.serving.mlx_shard import (
    ancestor_mask,
    decode_control,
    encode_compact,
    encode_tree_meta,
)


def test_control_frames_roundtrip_and_never_even() -> None:
    for depths, parents in [([0], [-1]), ([0, 1, 1, 2], [-1, 0, 0, 1])]:
        f = encode_tree_meta(depths, parents)
        assert len(f) % 2 == 1  # hidden frames are even; control must be odd
        assert decode_control(f) == ("tree", depths, parents)
    f = encode_compact([0, 2, 5])
    assert len(f) % 2 == 1
    assert decode_control(f) == ("compact", [0, 2, 5])


def test_decode_control_rejects_garbage() -> None:
    assert decode_control(b"\x00" * 9) is None       # odd length, bad magic
    assert decode_control(b"TREE") is None            # too short
    assert decode_control(b"\x00" * 8) is None        # even length


def test_ancestor_mask_chain_and_branch() -> None:
    # 0 -> 1 -> {2, 3}; 3 -> 4
    rows = ancestor_mask([-1, 0, 1, 1, 3])
    assert rows[0] == [True, False, False, False, False]
    assert rows[2] == [True, True, True, False, False]
    assert rows[3] == [True, True, False, True, False]   # sibling 2 masked out
    assert rows[4] == [True, True, False, True, True]


def test_tree_accept_walks_matching_children() -> None:
    # tree: 0 -> {1:t=10, 2:t=20}; 1 -> 3:t=30
    tokens = [5, 10, 20, 30]
    parents = [-1, 0, 0, 1]
    # target after node0 picks 10 (match 1), after node1 picks 30 (match 3),
    # after node3 picks 99 (no child) -> bonus
    tnext = {0: 10, 1: 30, 3: 99, 2: 0}
    accepted, bonus = tree_accept(tokens, parents, [tnext.get(i, 0) for i in range(4)])
    assert accepted == [0, 1, 3]
    assert bonus == 99


def test_tree_accept_root_reject_gives_bonus() -> None:
    accepted, bonus = tree_accept([5, 10], [-1, 0], [77, 0])
    assert accepted == [0]
    assert bonus == 77


def test_tree_draft_keep_interior_vs_leaf() -> None:
    # accepted path root->1->3 where 1 was fed at block col 0, 3 never fed (leaf)
    keep, last_unfed = tree_draft_keep([0, 1, 3], {1: 0})
    assert keep == [0]
    assert last_unfed is True
    # all accepted fed
    keep, last_unfed = tree_draft_keep([0, 1, 3], {1: 0, 3: 2})
    assert keep == [0, 2]
    assert last_unfed is False
    # root-only accept keeps nothing
    keep, last_unfed = tree_draft_keep([0], {})
    assert keep == []
    assert last_unfed is False


def test_engine_tree_param_validation() -> None:
    from soup.serving.mlx_engine import MLXClusterEngine

    with pytest.raises(ValueError, match="requires a draft"):
        MLXClusterEngine(model_id="m", remote_workers=[], tree_spec=True)
    with pytest.raises(ValueError, match="tree_budget"):
        MLXClusterEngine(model_id="m", remote_workers=[], draft_model_id="d",
                         tree_spec=True, tree_budget=1)
