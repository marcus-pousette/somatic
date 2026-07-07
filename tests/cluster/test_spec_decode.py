"""Unit tests for the speculative-decoding cache bookkeeping.

The distributed parts (draft loop, verify pass, wire trim frames) need MLX and a
live cluster; what must never regress silently is the trim arithmetic — the
verified invariant from the prototype: after every step, the target cache holds
exactly the accepted sequence and the draft cache sits one position behind the
next `cur`.
"""

import pytest

from soup.serving.mlx_engine import spec_trims


def test_full_accept_catches_draft_up() -> None:
    target_trim, draft_trim, catch_up = spec_trims(6, 6)
    assert target_trim == 0
    assert draft_trim == 0
    assert catch_up is True


@pytest.mark.parametrize("k", [1, 4, 6, 8])
def test_reject_path_invariants(k: int) -> None:
    for n_acc in range(k):
        target_trim, draft_trim, catch_up = spec_trims(k, n_acc)
        assert catch_up is False
        # Target consumed k+1 tokens (cur + k drafts), kept n_acc+1.
        assert target_trim == k - n_acc
        # Draft consumed cur + first k-1 drafts = k positions past the old cur;
        # after trimming it must sit just past the last accepted token.
        assert draft_trim == max(k - n_acc - 1, 0)
        draft_pos_after = k - draft_trim
        assert draft_pos_after == min(n_acc + 1, k)


def test_zero_accept_trims_everything_but_cur() -> None:
    target_trim, draft_trim, catch_up = spec_trims(4, 0)
    assert (target_trim, draft_trim, catch_up) == (4, 3, False)


def test_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        spec_trims(4, 5)
    with pytest.raises(ValueError):
        spec_trims(4, -1)


def test_num_draft_validated_at_construction() -> None:
    from soup.serving.mlx_engine import MLXClusterEngine

    with pytest.raises(ValueError, match="num_draft"):
        MLXClusterEngine(
            model_id="m", remote_workers=[], draft_model_id="d", num_draft=0
        )

