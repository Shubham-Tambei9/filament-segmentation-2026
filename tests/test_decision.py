"""Tests for the PQ-optimal decision layer (candidate tree + exact selection)."""
import itertools

import numpy as np

from filament_seg.decision import build_tree, labels_from_selection, select


def _random_forest(rng, n_levels=4, max_children=3):
    level, parent = [], []
    frontier = []
    for _ in range(rng.integers(1, 3)):  # roots
        level.append(0); parent.append(-1); frontier.append(len(level) - 1)
    for k in range(1, n_levels):
        nxt = []
        for v in frontier:
            for _ in range(rng.integers(0, max_children + 1)):
                level.append(k); parent.append(v); nxt.append(len(level) - 1)
        frontier = nxt
    return np.array(level), np.array(parent)


def _is_antichain(sel, parent):
    for v in np.flatnonzero(sel):
        a = parent[v]
        while a >= 0:
            if sel[a]:
                return False
            a = parent[a]
    return True


def test_select_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        level, parent = _random_forest(rng)
        if len(level) > 14:
            continue
        score = rng.random(len(level))
        tau = rng.uniform(0.2, 0.8)
        best = max(
            (sum(score[list(c)] - tau) if c else 0.0)
            for r in range(len(level) + 1)
            for c in itertools.combinations(range(len(level)), r)
            if _is_antichain(np.isin(np.arange(len(level)), c), parent)
        )
        sel = select(level, parent, score, tau)
        assert _is_antichain(sel, parent)
        assert np.isclose((score[sel] - tau).sum(), best)


def test_tree_is_nested_and_selection_is_disjoint():
    prob = np.zeros((128, 128), np.float32)
    prob[20:30, 10:60] = 0.5
    prob[22:28, 12:30] = 0.9  # two cores inside one faint filament
    prob[22:28, 40:58] = 0.9
    prob[80:90, 80:100] = 0.7
    image = np.full((128, 128), 120, np.uint8)
    tree, maps = build_tree(prob, image, levels=(0.3, 0.6, 0.8), keep_maps=True)
    for v in np.flatnonzero(tree.parent >= 0):  # every child lies inside its parent
        child = maps[tree.level[v]] == tree.local[v]
        par = tree.parent[v]
        assert np.all(maps[tree.level[par]][child] == tree.local[par])
    # high scores on the two cores, low on their union: the cores are chosen, not the union
    score = np.where(tree.level == 0, 0.2, 0.9)
    sel = select(tree.level, tree.parent, score, tau=0.3)
    labels = labels_from_selection(tree, maps, sel)
    assert labels.max() == sel.sum() >= 2
