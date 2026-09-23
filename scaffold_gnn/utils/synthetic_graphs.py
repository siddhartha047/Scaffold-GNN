"""Named synthetic graph families, and the long-range task that makes
congestion visible on them.

Why this module exists
----------------------
The objective ablation on the real benchmarks finds that dilation is the only
component of the certificate with a measurable downstream effect: with the
selector, the backbone and the budget held fixed, removing ``D`` from the
objective costs 0.66 pp while removing ``C_E`` or ``C_V`` costs nothing
(p >= 0.70). That is a statement about *those graphs*, not about the objective.
Cora, Chameleon, Squirrel, Amazon-* and Coauthor-CS are **route-rich**: between
two nodes there are many short alternatives, so a support that keeps detours
short incidentally keeps their load low, and pricing load separately buys
nothing.

These families are the complement. Each is **route-poor** somewhere by
construction, so a support can have short detours that all pile onto the same
few edges -- dilation stays small while congestion explodes. That is the regime
where the two congestion terms are supposed to do work, and the point of
registering the families as datasets is that the *same* ablation can be run on
them without a second code path.

  ``theta``  the extremal witness: k internally disjoint s-t paths. Keep one
             path and route the other k-1 detours through it and D = O(1) while
             C_E = Theta(k). Dilation cannot see the failure at all.
  ``grid``   the practical case, and the canonical over-squashing test bed: a
             4-neighbour lattice has long geodesics, uniform degree, and no cut
             to protect, so the only way to spend a budget badly is to
             concentrate load.
  ``torus``  the within-synthetic control: same degree and same n as ``grid``,
             but periodic, so every pair has strictly more routes. Congestion
             should matter *less* here than on ``grid`` at matched budget.
  ``bneck``  route scarcity as a dial: the min cut is exactly ``w``, so the
             congestion any support must accept on the surviving corridors
             scales like 1/w and the effect has to shrink as w grows -- which is
             what makes the claim falsifiable rather than anecdotal.
  ``sbm``    dense blocks, sparse cuts. Included because it is the family the
             pathology sweep already reports, so the ablation and that sweep can
             be read against each other.

The task (``anchor``)
---------------------
Congestion cannot matter for a task that a two-hop average already solves, so
these datasets do not ship a homophilous label. ``anchor_voronoi_task`` places
``n_anchors`` well-separated anchor nodes, marks each one with a one-hot channel
for its own identity, leaves every other node's features at zero, and labels
each non-anchor node with the anchor nearest to it in G. The label is therefore
recoverable *only* by moving anchor identity along paths, and a support that
routes many anchors' signals through one shared edge or node delivers a blend
that the receiver cannot decode -- over-squashing, in the form that
``C_E``/``C_V`` bound and ``D`` does not.

Two consequences to respect when using these datasets:

* **depth must reach.** The label at a node ``d`` hops from its anchor needs at
  least ``d`` message-passing layers. ``required_depth`` reports the anchor
  eccentricity for the built graph; a 2-layer model scores chance on
  ``synth-grid16`` no matter which support it is given, and that is a property
  of the model, not of the sparsifier.
* **anchors are excluded from every split**, by carrying label ``-1``: their own
  nearest anchor is themselves, so scoring them would hand every method free
  accuracy. ``rand_train_test_idx(..., ignore_negative=True)`` drops them.

Names
-----
As datasets, every spec is prefixed ``synth-`` and may carry an instance seed::

    synth-grid16          16x16 lattice, seed 0
    synth-grid16-s3       the same family, instance 3
    synth-torus16-s3      matched control for the above
    synth-bneck1 .. -s9   cut width 1, ten instances
    synth-theta14
    synth-sbm8-s2
    synth-barbell
    synth-ringclique8

``FAMILIES`` is the registry; ``describe(spec)`` returns the row for one spec so
a runner can print what it is about to measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# --- graph families ---------------------------------------------------------


def grid_graph(k: int = 16, seed: int = 0) -> nx.Graph:
    """k x k 4-neighbour lattice.

    ``seed`` is accepted and ignored: the lattice is deterministic. Instances of
    a grid family differ only in the task draw, which is what varying the seed
    of a ``synth-grid`` dataset does.
    """
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(k, k))


def torus_graph(k: int = 16, seed: int = 0) -> nx.Graph:
    """k x k periodic lattice: the matched control for ``grid_graph``.

    Same n, same 4-regular degree, but every node has a wrap-around route, so
    the number of short alternatives between a pair roughly doubles. Anything
    congestion-driven has to weaken here relative to the open grid; anything
    that survives unchanged was never about congestion.
    """
    return nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(k, k, periodic=True)
    )


def theta_graph(k: int = 14, path_len: int = 3, seed: int = 0) -> nx.Graph:
    """k internally disjoint s-t paths: the "dilation is not enough" witness.

    A support that keeps one path and drops the rest has D = path_len (short!)
    and C_E = k - 1 on every edge of the survivor. Dilation is blind to this by
    construction, which is why this family is the sanity check for any claim
    that the congestion terms are redundant.
    """
    G = nx.Graph()
    s, t = "s", "t"
    G.add_nodes_from([s, t])
    for i in range(k):
        prev = s
        for j in range(path_len - 1):
            node = (i, j)
            G.add_edge(prev, node)
            prev = node
        G.add_edge(prev, t)
    return nx.convert_node_labels_to_integers(G)


def bottleneck_graph(width: int = 4, blob: int = 50, blob_degree: int = 10,
                     corridor_len: int = 3, seed: int = 0) -> nx.Graph:
    """Two dense blobs joined by exactly ``width`` disjoint corridors.

    The min edge cut between the blobs is exactly ``width`` for every
    ``width <= blob_degree``, by construction. The blobs are random *regular*
    rather than Erdos-Renyi so that min degree is pinned at ``blob_degree`` and
    the corridors stay the binding cut -- with ER blobs the cheapest cut becomes
    "isolate one low-degree blob node" once ``width`` exceeds the min degree,
    and the dial silently saturates.

    This is the same generator the pathology sweep uses; see
    ``scripts/experiments/pathologies/run_pathology_structural.py`` for the
    longer design note and for the two variants that did not work.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if width > blob_degree:
        raise ValueError(
            f"width {width} exceeds blob_degree {blob_degree}: the min cut "
            f"would be a low-degree blob node, not the corridors, and the "
            f"width dial would no longer control the bottleneck"
        )
    rng = np.random.default_rng(seed)
    A = nx.random_regular_graph(blob_degree, blob, seed=int(rng.integers(1 << 31)))
    B = nx.random_regular_graph(blob_degree, blob, seed=int(rng.integers(1 << 31)))
    G = nx.Graph()
    G.add_nodes_from(("a", v) for v in A.nodes())
    G.add_edges_from((("a", u), ("a", v)) for u, v in A.edges())
    G.add_nodes_from(("b", v) for v in B.nodes())
    G.add_edges_from((("b", u), ("b", v)) for u, v in B.edges())
    ends_a = rng.choice(blob, size=width, replace=False)
    ends_b = rng.choice(blob, size=width, replace=False)
    for i, (ua, ub) in enumerate(zip(ends_a, ends_b)):
        prev = ("a", int(ua))
        for j in range(corridor_len - 1):
            node = ("c", i, j)
            G.add_edge(prev, node)
            prev = node
        G.add_edge(prev, ("b", int(ub)))
    return nx.convert_node_labels_to_integers(G)


def sbm_graph(blocks: int = 8, size: int = 25, p_in: float = 0.50,
              p_out: float = 0.004, seed: int = 0) -> nx.Graph:
    """Planted partition: dense blocks joined by sparse cuts.

    Restricted to the giant component so the spanning-forest floor is well
    defined and the unreachable-pair metrics are not dominated by pre-existing
    debris. Same parameters as the pathology sweep's ``sbm`` family.
    """
    probs = [[p_in if i == j else p_out for j in range(blocks)]
             for i in range(blocks)]
    G = nx.Graph(nx.stochastic_block_model([size] * blocks, probs, seed=seed))
    G.remove_edges_from(nx.selfloop_edges(G))
    giant = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(giant).copy())


def barbell_graph(clique: int = 20, path_len: int = 6, seed: int = 0) -> nx.Graph:
    """Two cliques joined by a path: min cut 1, so every non-backbone method
    has a standing chance of disconnecting the graph outright."""
    return nx.convert_node_labels_to_integers(
        nx.barbell_graph(clique, path_len)
    )


def ring_of_cliques(cliques: int = 8, clique: int = 5, seed: int = 0) -> nx.Graph:
    """Local redundancy plus one global cycle: the cheap detour around a
    removed clique edge is inside the clique, but the cheap detour around a
    removed ring edge is all the way round."""
    return nx.convert_node_labels_to_integers(
        nx.ring_of_cliques(cliques, clique)
    )


def hub_block_ring(block: int = 16, blocks: int = 6, seed: int = 0) -> nx.Graph:
    """Ring of *large* cliques: the family where dilation and node congestion both
    have room, which no other family here manages.

    This is ``ring_of_cliques`` with the block size promoted to the swept parameter,
    and the reason it is registered separately is that the block size is the dial
    that decides whether ``C_V`` is a different quantity from ``C_E``:

    * inside a block, every spanning tree costs the same ``block - 1`` edges, and a
      *star* is one of them. So the support can put a degree-``block`` hub at the
      centre of every block **for free**, and then all ``C(block,2) - block + 1``
      omitted block edges route ``u - c - v``: one intermediate vertex against the
      centre's ``block - 1`` spokes, so ``C_V / C_E ~ block / 2``.
    * a path spanning the same block costs the same ``block - 1`` edges and gives
      ``D = block`` with small ``C_V``. So star-versus-path is a real
      dilation-against-node-congestion decision taken at *no budget cost*.
    * between blocks the ring means dropping a bridge costs a detour all the way
      round, so global ``D`` keeps a wide range across arms.

    Measured by ``Brainstrom/ablations/sketch/screen_roc_tension.py``, and the two
    dials are independent -- which is the point, since on every other family they
    are not:

        block  5 -> C_V/C_E 2.0     block 10 -> 4.5     block 16 -> 7.5
        blocks 6 -> D range 2..12   blocks 10 -> D range 2..20   (at every block size)

    The failure mode ``C_V`` was introduced for is visible in the certificates at
    ``block = 16``: the dilation-only arm reaches ``C_V = 105``, the worst of all
    eight arms, and adding the ``C_V`` term pulls it to 90 (eta 0.15) or 74
    (eta 0.30). Contrast ``hub_ring_graph``, where the hub has to be *bought* out of
    the discretionary budget and the ratio therefore never exceeds 2.2, and
    ``spider_graph``, where the hub is free but universal so every arm achieves
    ``D = 2`` and dilation has nothing to choose.
    """
    if block < 3 or blocks < 3:
        raise ValueError("need block >= 3 and blocks >= 3")
    return nx.convert_node_labels_to_integers(nx.ring_of_cliques(blocks, block))


def spider_graph(corridors: int = 12, path_len: int = 6, seed: int = 0) -> nx.Graph:
    """``corridors`` disjoint paths plus one universal hub: the C_V witness.

    This is to node congestion what ``theta_graph`` is to edge congestion. Keep
    only the hub's spokes and every omitted corridor edge ``(u, v)`` routes
    ``u - h - v``, so

        D = 2                          (as short as any detour can be)
        C_E <= 2                       (a spoke carries only its own endpoint's
                                        two corridor edges)
        C_V(h) = corridors * (path_len - 1)

    -- ``C_V`` grows with the whole graph while ``D`` and ``C_E`` stay constant,
    so the achievable ``C_V / C_E`` ratio is Theta(n) rather than the ``d_max/2``
    that bounds it on every bounded-degree family. Keeping the corridor edges
    instead costs ``D = path_len`` and gives ``C_V ~ path_len``: the opposite
    corner. The two extremes need the *same* number of edges
    (``corridors * path_len`` either way), so the choice between them is not a
    budget artefact.

    Note that a universal hub collapses the diameter to 2, so this family cannot
    carry a *distance* task -- it is a certificate witness and a retrieval-task
    family, not an anchor-task family.
    """
    if corridors < 2 or path_len < 3:
        raise ValueError("need corridors >= 2 and path_len >= 3")
    G = nx.Graph()
    hub = "h"
    G.add_node(hub)
    for i in range(corridors):
        nodes = [(i, j) for j in range(path_len)]
        nx.add_path(G, nodes)
        for v in nodes:
            G.add_edge(hub, v)
    return nx.convert_node_labels_to_integers(G)


def hub_ring_graph(chords: int = 60, ring: int = 120, spoke_stride: int = 4,
                   seed: int = 0) -> nx.Graph:
    """Ring with long chords, plus a moderate-degree hub that shortcuts it.

    A first attempt at making dilation and node congestion conflict, kept because
    its *failure* is the diagnosis that produced ``hub_block_ring``, which is the
    family to use. Two requirements have to hold at once and they pull against each
    other:

    * **``D`` must have room.** A chord's endpoints are at least ``ring // 4``
      apart on the ring, so dropping a chord and detouring around the ring costs
      O(ring) hops -- dilation has real work to do, unlike on a clique pair where
      every arm already achieves ``D = 2``.
    * **``C_V`` must have room.** That needs a high-degree vertex in the support,
      which needs one in the graph.

    The hub supplies both at once, and that is the point: it is adjacent to every
    ``spoke_stride``-th ring node, so *any* chord can be replaced by a 4-5 hop
    ride through the hub. Dilation therefore prefers to spend its budget on
    spokes and abandon the chords -- which funnels every omitted chord through one
    vertex and sends ``C_V`` to Theta(chords) while ``C_E`` per spoke stays at
    about ``chords / (ring / spoke_stride)``. Node congestion is the only term in
    the objective that objects. That is the hub-formation failure mode ``C_V`` was
    introduced for, expressed as a family rather than as an argument.

    The hub is deliberately *not* universal (degree ``ring / spoke_stride``, not
    ``ring``): a universal hub is a neighbour of everything, so its mixed
    representation poisons the full graph too and there is no headroom left to
    measure. ``spoke_stride=1`` is allowed but is that degenerate corner -- it is
    only there as the endpoint of the tension sweep in
    ``Brainstrom/ablations/sketch/screen_hub_tension.py``.

    What actually happens, and why it matters. That sweep put ``spoke_stride`` through
    1, 2, 3, 4, 6, 8, 12 and the achieved ``C_V / C_E`` never exceeded **2.24** at any
    of them -- nowhere near the ``deg_max(G) / 2 = 60`` the construction was aiming
    for. The reason is the budget, not the geometry: the hub is *not* in the spanning
    forest, so every spoke the support keeps is a discretionary edge it did not spend
    on a chord. ``deg_max(H)`` therefore stalls at 7-55 against ``deg_max(G)`` of
    10-120, and since ``C_V / C_E ~ deg_max(H) / 2``, the ratio stalls with it. The
    design rule that follows -- **the hub has to be free** -- is what
    ``hub_block_ring`` implements, and it reaches 7.5 with the same ``D`` range.
    """
    if ring < 16 or spoke_stride < 1:
        raise ValueError("need ring >= 16 and spoke_stride >= 1")
    G = nx.cycle_graph(ring)
    rng = np.random.default_rng(seed)
    far = max(2, ring // 4)
    placed = 0
    guard = 0
    while placed < chords and guard < 100 * max(chords, 1):
        guard += 1
        u, v = (int(x) for x in rng.choice(ring, size=2, replace=False))
        gap = abs(u - v)
        if min(gap, ring - gap) < far or G.has_edge(u, v):
            continue
        G.add_edge(u, v)
        placed += 1
    if placed < chords:
        raise RuntimeError(
            f"only placed {placed} of {chords} long chords on a ring of "
            f"{ring}; lower `chords` or raise `ring`"
        )
    hub = ring
    for v in range(0, ring, spoke_stride):
        G.add_edge(hub, v)
    return nx.convert_node_labels_to_integers(G)


@dataclass(frozen=True)
class Family:
    """One registered family: how to build it, and what it is for."""

    name: str
    builder: object
    param: str                     # name of the integer knob in the spec
    default: int
    anchors: int                   # default number of classes for the task
    note: str
    aliases: tuple = field(default=())


FAMILIES = {
    "grid": Family(
        "grid", grid_graph, "k", 16, 8,
        "k x k lattice; MEASURED route-rich -- it was expected to be the "
        "congestion case and is not: on a distance task the congestion terms "
        "cost 1.9-4.9 pp there because they compete with dilation for the same "
        "budget",
        aliases=("lattice",),
    ),
    "torus": Family(
        "torus", torus_graph, "k", 16, 8,
        "k x k periodic lattice; matched control for grid -- same n and degree, "
        "strictly more routes",
    ),
    "theta": Family(
        "theta", theta_graph, "k", 14, 2,
        "k internally disjoint s-t paths; extremal witness with D = O(1) and "
        "C_E = Theta(k)",
    ),
    "bneck": Family(
        "bneck", bottleneck_graph, "width", 4, 2,
        "two 10-regular blobs of 50 joined by exactly `width` disjoint "
        "corridors; route scarcity as a dial",
        aliases=("bottleneck",),
    ),
    "sbm": Family(
        "sbm", sbm_graph, "blocks", 8, 8,
        "planted partition, dense blocks and sparse cuts",
    ),
    "barbell": Family(
        "barbell", barbell_graph, "clique", 20, 2,
        "two cliques joined by a path; min cut 1",
    ),
    "ringclique": Family(
        "ringclique", ring_of_cliques, "cliques", 8, 8,
        "ring of cliques; local redundancy plus one global cycle",
        aliases=("tree_cycle", "ring_of_cliques"),
    ),
    "spider": Family(
        "spider", spider_graph, "corridors", 12, 2,
        "corridors plus a universal hub; extremal C_V witness with D = 2, "
        "C_E = O(1) and C_V = Theta(n) -- the node-congestion analogue of "
        "theta. Diameter 2, so retrieval only, never anchor",
        aliases=("corridor_hub",),
    ),
    "hubring": Family(
        "hubring", hub_ring_graph, "chords", 60, 2,
        "ring with long chords plus a moderate-degree hub shortcut; the only "
        "family where dilation and C_V actually conflict -- dilation prefers to "
        "buy spokes and abandon the chords, which is exactly the hub formation "
        "C_V is supposed to prevent",
        aliases=("ringhub",),
    ),
    "hubblock": Family(
        "hubblock", hub_block_ring, "block", 16, 2,
        "ring of large cliques, swept on block size; the only family with both a "
        "wide D range and real C_V headroom, because a block's spanning star is "
        "free -- C_V/C_E ~ block/2 while the ring keeps D in play",
        aliases=("starblock", "ring_of_big_cliques"),
    ),
}

_ALIAS = {alias: fam.name for fam in FAMILIES.values() for alias in fam.aliases}

_SPEC = re.compile(r"^([a-z_]+?)(\d*)(?:-s(\d+))?$")


def parse_spec(spec: str) -> tuple[Family, int, int]:
    """``'grid16-s3'`` -> (grid family, k=16, seed=3).

    The prefix ``synth-`` is optional so that the same string works as a
    dataset name and as a bare family spec.
    """
    text = str(spec).strip().lower()
    for prefix in ("synth-", "synthetic-", "synth_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    match = _SPEC.match(text)
    if match is None:
        raise ValueError(
            f"cannot parse synthetic spec {spec!r}; expected "
            f"<family>[<param>][-s<seed>], e.g. grid16-s3"
        )
    name, param, seed = match.groups()
    name = _ALIAS.get(name, name)
    if name not in FAMILIES:
        raise ValueError(
            f"unknown synthetic family {name!r}; registered: "
            f"{', '.join(sorted(FAMILIES))}"
        )
    family = FAMILIES[name]
    return family, int(param) if param else family.default, int(seed or 0)


def build_graph(spec: str) -> nx.Graph:
    """Build one instance. Node labels are always ``0..n-1``."""
    family, param, seed = parse_spec(spec)
    G = family.builder(**{family.param: param}, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def describe(spec: str) -> dict:
    """What a runner should print before measuring this instance."""
    family, param, seed = parse_spec(spec)
    G = build_graph(spec)
    x, y, anchors = anchor_voronoi_task(G, family.anchors, seed=seed)
    return {
        "spec": f"{family.name}{param}-s{seed}",
        "family": family.name,
        family.param: param,
        "seed": seed,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "n_classes": int(len(anchors)),
        "required_depth": required_depth(G, anchors),
        "forest_floor": (G.number_of_nodes() - nx.number_connected_components(G))
        / max(1, G.number_of_edges()),
        "note": family.note,
    }


# --- the long-range task ----------------------------------------------------


def _farthest_point_anchors(G, n_anchors: int, seed: int) -> list[int]:
    """Greedy farthest-point sampling from a seeded random start.

    Uniform random anchors on a lattice give badly unbalanced Voronoi cells --
    two anchors three hops apart split ~6 nodes between them while a third owns
    the rest -- and an unbalanced label makes a majority predictor look good,
    which is exactly the artefact that would swamp a support comparison.
    Farthest-point sampling is deterministic given the start, so an instance
    seed still gives independent draws.
    """
    nodes = sorted(G.nodes())
    rng = np.random.default_rng(int(seed))
    first = int(nodes[rng.integers(len(nodes))])
    anchors = [first]
    dist = np.array([
        nx.single_source_shortest_path_length(G, first).get(v, np.inf)
        for v in nodes
    ], dtype=float)
    while len(anchors) < min(n_anchors, len(nodes)):
        pick = int(np.argmax(np.where(np.isfinite(dist), dist, -1.0)))
        if dist[pick] <= 0:
            break
        anchors.append(int(nodes[pick]))
        d_new = nx.single_source_shortest_path_length(G, nodes[pick])
        dist = np.minimum(dist, np.array(
            [d_new.get(v, np.inf) for v in nodes], dtype=float))
    return anchors


def anchor_voronoi_task(G, n_anchors: int, seed: int = 0,
                        marker_scale: float = 1.0, noise_dim: int = 0,
                        noise_scale: float = 0.1):
    """Label every node by its nearest anchor; mark only the anchors.

    Returns ``(x, y, anchors)`` with ``x`` of shape ``(n, n_anchors + noise_dim)``
    and ``y`` of shape ``(n,)``. Anchors carry ``y = -1`` so the split helper
    drops them: their own nearest anchor is themselves, and scoring that would
    hand every support free accuracy.

    Nodes that no anchor can reach also carry ``-1``. That only happens on a
    disconnected input; on a support it cannot happen, because the task is
    always built from ``G``.
    """
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    anchors = _farthest_point_anchors(G, n_anchors, seed)
    k = len(anchors)

    best = np.full(n, np.inf)
    label = np.full(n, -1, dtype=np.int64)
    for a_idx, a in enumerate(anchors):
        d = nx.single_source_shortest_path_length(G, a)
        for v, dv in d.items():
            i = index[v]
            if dv < best[i]:
                best[i] = dv
                label[i] = a_idx

    rng = np.random.default_rng(int(seed) + 9973)
    x = np.zeros((n, k + int(noise_dim)), dtype=np.float32)
    for a_idx, a in enumerate(anchors):
        x[index[a], a_idx] = float(marker_scale)
    if noise_dim:
        x[:, k:] = rng.normal(0.0, float(noise_scale),
                              size=(n, int(noise_dim))).astype(np.float32)

    for a in anchors:                      # anchors are not scored
        label[index[a]] = -1
    label[~np.isfinite(best)] = -1
    return x, label, anchors


def required_depth(G, anchors) -> int:
    """Hops the deepest scored node needs before its label is reachable."""
    nodes = sorted(G.nodes())
    best = np.full(len(nodes), np.inf)
    index = {v: i for i, v in enumerate(nodes)}
    for a in anchors:
        for v, dv in nx.single_source_shortest_path_length(G, a).items():
            best[index[v]] = min(best[index[v]], dv)
    finite = best[np.isfinite(best)]
    return int(finite.max()) if finite.size else 0


def global_sign_task(G, n_sources: int = 16, seed: int = 0,
                     bit_scale: float = 10.0):
    """Every scored node predicts the sign of the sum of all source bits.

    Returns ``(x, y, sources)`` with ``x`` of shape ``(n, 2)`` -- column 0 the
    signed bit scaled by ``bit_scale``, column 1 a source marker -- and ``y`` of
    shape ``(n,)``. Sources carry ``y = -1``: they hold a piece of the answer as a
    local feature, and scoring them would reward reading rather than routing.

    Why this exists next to ``anchor_voronoi_task``
    -----------------------------------------------
    Nearest-anchor is a *distance* question. A node needs only the identity of its
    closest marker, which is decidable inside a ball of radius D, so any support
    that keeps detours short answers it and load never enters the problem. That
    makes it a dilation task by construction, and it is why the {D, C_E, C_V} cube
    run on a grid with the anchor task is silent about congestion: there was no
    mechanism for congestion to act through.

    Here the label is a function of *all* ``n_sources`` bits jointly, so every
    scored node needs every source. On a support with few routes those n_sources
    signal paths must share the same edges, and spreading that load is exactly
    what the C_E and C_V terms buy. Congestion can only bind on a task of this
    second kind.

    The cost of that design: the label is one bit per instance, shared by every
    scored node, so the effective sample size is the number of tiled instances,
    not the number of nodes. A per-node label computable from a neighbourhood
    would be a distance task again, so this is not a fixable shortcoming -- tile
    to a few hundred instances instead.

    ``sum`` is taken over an odd count so the sign is never zero; ``n_sources`` is
    forced odd for that reason, and a graph with fewer nodes than that gets as
    many sources as it can hold.
    """
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    k = min(int(n_sources), n - 1 if n > 1 else 1)
    if k % 2 == 0:                         # keep the sign strictly non-zero
        k -= 1
    k = max(k, 1)

    rng = np.random.default_rng(int(seed) + 104729)
    sources = [nodes[i] for i in rng.choice(n, size=k, replace=False)]
    bits = rng.choice(np.array([-1, 1]), size=k)

    x = np.zeros((n, 2), dtype=np.float32)
    for s, b in zip(sources, bits):
        x[index[s], 0] = float(bit_scale) * float(b)
        x[index[s], 1] = float(bit_scale)

    label = np.full(n, 1 if int(bits.sum()) > 0 else 0, dtype=np.int64)
    for s in sources:
        label[index[s]] = -1
    return x, label, sources


def aggregation_depth(G, sources) -> int:
    """Hops the deepest scored node needs before *every* source is reachable.

    The max over sources, not the min: unlike the anchor task, a node that has
    heard from the nearest source still cannot answer. This is close to the graph
    diameter, which is why the aggregation task is only affordable on small
    families -- a 16x16 grid would want ~30 layers.
    """
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    worst = np.zeros(len(nodes))
    for s in sources:
        d = nx.single_source_shortest_path_length(G, s)
        reached = np.full(len(nodes), np.inf)
        for v, dv in d.items():
            reached[index[v]] = dv
        worst = np.maximum(worst, reached)
    finite = worst[np.isfinite(worst)]
    return int(finite.max()) if finite.size else 0


def keyed_retrieval_task(G, n_sources: int = 15, seed: int = 0,
                         bit_scale: float = 10.0):
    """Every scored node must retrieve the bit of *one named* remote source.

    Returns ``(x, y, sources)`` with ``x`` of shape ``(n, 2k + 1)``: columns
    ``0:k`` a one-hot saying "I am source j", column ``k`` that source's signed
    bit, columns ``k+1:2k+1`` a one-hot query saying "answer with source j's
    bit". Sources carry ``y = -1``; a scored node's label is the bit of the
    source its query names.

    Why this exists next to ``global_sign_task``
    -------------------------------------------
    The sign task is an aggregation task and so congestion can act on it -- but
    only *edge* congestion. Its label is a symmetric sum, and a sum is precisely
    what a mean- or sum-aggregating layer computes for free: a degree-d hub can
    forward the total of its d neighbours in one scalar with no loss, so routing
    every signal path through one vertex costs nothing. That makes the sign task
    structurally blind to node congestion, in the same way the anchor task is
    structurally blind to congestion of either kind. The measured C_V nulls on the
    bottleneck and lattice cubes are consistent with the term being useless and
    equally consistent with no task in the suite being able to see it.

    Retrieval breaks that. Here the k source bits are *distinguishable items*: a
    node's answer depends on exactly one of them, chosen by a key it carries in
    its own features, so nothing can be summarised away en route. A hub of degree
    d that carries all k signal paths must keep k keyed bits separable in one
    fixed-width vector, and a normalised aggregator attenuates each contribution
    by about 1/d on the way in. That penalty is a function of the hub's degree
    alone -- exactly the quantity ``support_stats`` charges to C_V -- and it is
    absent for a sum. So a support that spreads load over many vertices should
    beat one that funnels it through a few, and the C_V term has a mechanism.

    Dilation still matters, because a bit that cannot arrive within the layer
    budget is unreadable, so this task does not isolate C_V -- the cube does that.
    What it does is make C_V *visible*, which the other two tasks cannot.
    """
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    k = int(min(n_sources, max(n // 2, 1)))     # leave half the graph to score
    rng = np.random.default_rng(int(seed) + 15485863)
    sources = [nodes[i] for i in rng.choice(n, size=k, replace=False)]
    bits = rng.integers(0, 2, size=k)
    source_set = set(sources)

    x = np.zeros((n, 2 * k + 1), dtype=np.float32)
    for j, (s, b) in enumerate(zip(sources, bits)):
        x[index[s], j] = float(bit_scale)
        x[index[s], k] = float(bit_scale) * (2.0 * float(b) - 1.0)

    label = np.full(n, -1, dtype=np.int64)
    queries = {}
    for v in nodes:
        if v in source_set:
            continue
        j = int(rng.integers(0, k))
        x[index[v], k + 1 + j] = float(bit_scale)
        label[index[v]] = int(bits[j])
        queries[v] = sources[j]
    return x, label, sources


def retrieval_depth(G, sources, queries=None) -> int:
    """Hops before every scored node can reach the source its query names.

    Without ``queries`` this is the same worst case as ``aggregation_depth``: any
    node may be asked about any source, so the layer budget has to cover the
    farthest one. Passing the realised assignment gives the tighter number, but
    the loose bound is what the runner uses, because the assignment is redrawn
    per instance and the model depth is fixed across instances.
    """
    if queries is None:
        return aggregation_depth(G, sources)
    worst = 0
    for v, s in queries.items():
        try:
            worst = max(worst, nx.shortest_path_length(G, v, s))
        except nx.NetworkXNoPath:
            continue
    return int(worst)
