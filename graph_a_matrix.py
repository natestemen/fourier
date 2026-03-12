"""
Graph-theoretic exploration of the A-matrix structure.

For a partition λ, form a bipartite graph:
  - Left nodes:  addable cells  aᵢ  (labeled by content c(aᵢ) = x-y)
  - Right nodes: removable cells rⱼ (labeled by content c(rⱼ))
  - Edges:       aᵢ -- rⱼ  with weight  c(aᵢ) - c(rⱼ)   (the denominator in A[i,j])

Interesting questions:
  1. Do the contents interlace on the number line?
  2. Is the weight matrix a Cauchy matrix?
  3. What is the spectral structure of the weighted graph?
  4. How does the graph structure determine the A-matrix?
"""

import sympy as sp
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from symbolic_a_matrix import (
    build_symbolic_a_matrix_for_partition,
    _partition_addable_cells,
    _partition_removable_cells,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def content(cell):
    x, y = cell
    return int(x - y)


def build_content_graph(partition):
    """Return (G, addable_contents, removable_contents) for the partition."""
    add = _partition_addable_cells(partition)
    rem = _partition_removable_cells(partition)
    ac = [content(a) for a in add]
    rc = [content(r) for r in rem]

    G = nx.Graph()
    for c in ac:
        G.add_node(f"a{c:+d}", side="add", content=c)
    for c in rc:
        G.add_node(f"r{c:+d}", side="rem", content=c)
    for ca in ac:
        for cr in rc:
            G.add_edge(f"a{ca:+d}", f"r{cr:+d}", weight=ca - cr, abs_weight=abs(ca - cr))
    return G, ac, rc


def cauchy_det_formula(xs, ys):
    """Cauchy determinant: ∏(xᵢ-xⱼ)∏(yᵢ-yⱼ) / ∏ᵢⱼ(xᵢ-yⱼ), for square case."""
    n = len(xs)
    num = sp.Integer(1)
    for i in range(n):
        for j in range(i+1, n):
            num *= (xs[i] - xs[j]) * (ys[i] - ys[j])
    den = sp.Integer(1)
    for x in xs:
        for y in ys:
            den *= (x - y)
    return num / den


# ── analysis functions ─────────────────────────────────────────────────────────

def analyse_partition(partition, verbose=True):
    label = str(tuple(partition))
    G, ac, rc = build_content_graph(partition)
    A = build_symbolic_a_matrix_for_partition(partition)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Partition {label}")
        print(f"{'═'*60}")
        print(f"  Addable   contents (sorted): {sorted(ac, reverse=True)}")
        print(f"  Removable contents (sorted): {sorted(rc, reverse=True)}")

        # ── 1. Interlacing ────────────────────────────────────────────────
        merged = sorted([(c, 'A') for c in ac] + [(c, 'R') for c in rc], reverse=True)
        seq = ' > '.join(f"{c}({'A' if t=='A' else 'R'})" for c, t in merged)
        interlaces = all(
            merged[i][1] != merged[i+1][1] for i in range(len(merged)-1)
        )
        print(f"\n  1. INTERLACING on the content line:")
        print(f"     {seq}")
        print(f"     Strict interlacing (A,R alternate): {interlaces}")

        # ── 2. Weight (distance) matrix ──────────────────────────────────
        ac_s = sorted(ac, reverse=True)
        rc_s = sorted(rc, reverse=True)
        W = np.array([[ca - cr for cr in rc_s] for ca in ac_s])
        print(f"\n  2. WEIGHT MATRIX  W[i,j] = c(aᵢ) - c(rⱼ)  (all positive due to interlacing):")
        col_hdr = "        " + "  ".join(f"r{c:+d}" for c in rc_s)
        print(f"     {col_hdr}")
        for i, ca in enumerate(ac_s):
            row = "  ".join(f"   {W[i,j]:2d}" for j in range(len(rc_s)))
            print(f"     a{ca:+d} |{row}")

            # With interlacing: W[i,j] > 0 iff i <= j (upper-left block positive)
        upper_pos = all(W[i,j] > 0 for i in range(len(ac_s)) for j in range(len(rc_s)) if i <= j)
        lower_neg = all(W[i,j] < 0 for i in range(len(ac_s)) for j in range(len(rc_s)) if i > j)
        print(f"     Sign pattern (W[i,j]>0 iff i≤j): upper={upper_pos}, lower={lower_neg}")

        # ── 3. Cauchy matrix ──────────────────────────────────────────────
        # The non-dummy A-matrix columns have denominator 1/W[i,j]
        # Factor out the row-weights (sqrt terms) → get a matrix ∝ 1/W
        # Check: is the submatrix A[:,1:] a "scaled Cauchy matrix"?
        A_num = np.array(A.tolist(), dtype=float)
        # Extract non-dummy part (cols 1..)
        B = A_num[:, 1:]      # shape (k, k-1)
        # Compare with 1/W
        if B.shape[1] > 0:
            ratio = B * W  # should be constant per row if Cauchy structure
            print(f"\n  3. CAUCHY STRUCTURE:  A[i,j] = row_weight[i] / W[i,j]")
            print(f"     B*W  (should be constant per row):")
            for i, ca in enumerate(ac_s):
                row_vals = "  ".join(f"{ratio[i,j]:7.4f}" for j in range(len(rc_s)))
                print(f"     a{ca:+d} | {row_vals}")
            # A[i,j] = α_i * β_j / W[i,j]  →  A[i,j]*W[i,j] = α_i * β_j  (rank-1 outer product)
            # Check rank of B*W (element-wise):
            BW = B * W
            rank = np.linalg.matrix_rank(BW, tol=1e-9)
            row_const = np.allclose(ratio, ratio[:, [0]], atol=1e-9)
            print(f"     Rows constant (β_j all equal): {row_const}")
            print(f"     rank(B∘W) = {rank}  (should be 1 if α_i·β_j outer product)")

        # ── 4. Graph spectral properties ──────────────────────────────────
        # Build weighted adjacency matrix of bipartite graph
        n_add, n_rem = len(ac), len(rc)
        # Use 1/weight as the "strength" (larger distance = weaker connection)
        inv_W = 1.0 / W  # shape (n_add, n_rem)
        # Full adjacency in node order [addable..., removable...]
        N = n_add + n_rem
        Adj = np.zeros((N, N))
        Adj[:n_add, n_add:] = inv_W
        Adj[n_add:, :n_add] = inv_W.T
        eigvals = np.linalg.eigvalsh(Adj)
        eigvals_sorted = sorted(eigvals, key=abs, reverse=True)
        print(f"\n  4. SPECTRAL STRUCTURE of bipartite graph (weights=1/distance):")
        print(f"     Eigenvalues: {[f'{e:.4f}' for e in eigvals_sorted]}")
        print(f"     Spectral gap (|λ₁|-|λ₂|): {abs(eigvals_sorted[0])-abs(eigvals_sorted[1]):.4f}")
        print(f"     Zero eigenvalues: {sum(abs(e)<1e-10 for e in eigvals)}")

        # ── 5. Determinant of 1/W (Cauchy det) ───────────────────────────
        if n_add == n_rem + 1:  # square after dropping one row
            # The "core" Cauchy-like matrix: drop first addable → square (n_rem x n_rem)
            # But more interesting: det of W itself for the square submatrix
            xs = sp.Matrix(sorted(ac, reverse=True)[:-1])  # drop smallest addable
            ys = sp.Matrix(sorted(rc, reverse=True))
            W_sq = sp.Matrix([[a - b for b in ys] for a in xs])
            d = sp.simplify(W_sq.det())
            print(f"\n  5. Det of square sub-weight-matrix (drop smallest addable):")
            print(f"     {d}")

        # ── 6. Min spanning tree edges ────────────────────────────────────
        G2 = nx.Graph()
        for i, ca in enumerate(ac_s):
            for j, cr in enumerate(rc_s):
                G2.add_edge(f"a{ca:+d}", f"r{cr:+d}", weight=W[i,j])
        mst = nx.minimum_spanning_tree(G2)
        print(f"\n  6. MINIMUM SPANNING TREE edges (min-weight = closest content pairs):")
        for u, v, d in sorted(mst.edges(data=True), key=lambda e: e[2]['weight']):
            print(f"     {u} -- {v}  (distance={d['weight']})")

    return G, ac, rc, A


# ── run for all 3-addable partitions up to level 6 ────────────────────────────

test_partitions = [
    [1],          # 2-addable (level 1)
    [2],          # 2-addable
    [1,1],        # 2-addable
    [2,1],        # 3-addable (level 3)
    [3,1],        # 3-addable (level 4)
    [2,1,1],      # 3-addable (level 4)
    [3,2],        # 3-addable (level 5)
    [4,1],        # 3-addable (level 5)
    [3,1,1],      # 3-addable (level 5)
    [2,2,1],      # 3-addable (level 5)
    [3,2,1],      # 4-addable (level 6) — for comparison
]

for p in test_partitions:
    analyse_partition(p)


# ── global observation: do the weights only ever take values {1,2,3,...}? ─────
print(f"\n\n{'═'*60}")
print("  GLOBAL: Weight values across all partitions (up to level 8)")
print(f"{'═'*60}")
from helper import partitions
all_weights = set()
for n in range(1, 9):
    for p in partitions(n):
        add = _partition_addable_cells(p)
        rem = _partition_removable_cells(p)
        for a in add:
            for r in rem:
                all_weights.add(content(a) - content(r))
print(f"  Possible weight values: {sorted(all_weights)}")
print(f"  All strictly positive: {all(w > 0 for w in all_weights)}")


# ── visualise: content-line diagram for the 3-addable family ──────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle("A-matrix bipartite content graphs\n(nodes = content values, edges weighted by distance)", fontsize=13)
axes = axes.flatten()

three_addable = [[2,1],[3,1],[2,1,1],[4,1],[3,2],[3,1,1],[2,2,1],[3,2,1]]

for ax, partition in zip(axes, three_addable):
    G, ac, rc = build_content_graph(partition)
    ac_s = sorted(ac, reverse=True)
    rc_s = sorted(rc, reverse=True)

    # Layout: addable on top row, removable on bottom row, x = content value
    all_c = sorted(set(ac + rc))
    xmin, xmax = min(all_c) - 0.5, max(all_c) + 0.5
    pos = {}
    for c in ac:
        pos[f"a{c:+d}"] = (c, 1)
    for c in rc:
        pos[f"r{c:+d}"] = (c, 0)

    # Draw edges, thickness ∝ 1/distance
    max_w = max(abs(ca - cr) for ca in ac for cr in rc)
    for ca in ac_s:
        for cr in rc_s:
            d = ca - cr
            lw = max(0.5, 3.5 * (max_w - d + 1) / max_w)
            alpha = min(1.0, max(0.1, 0.3 + 0.6 * (max_w - d + 1) / max_w))
            ax.plot([ca, cr], [1, 0], color='steelblue', lw=lw, alpha=alpha, zorder=1)
            mid_x = (ca + cr) / 2
            ax.text(mid_x, 0.5, str(d), ha='center', va='center',
                    fontsize=6.5, color='darkblue', bbox=dict(fc='white', ec='none', pad=0.5))

    # Draw nodes
    for c in ac:
        ax.scatter(c, 1, s=200, color='tomato', zorder=3)
        ax.text(c, 1.18, f"a{c:+d}", ha='center', va='bottom', fontsize=8, color='tomato', fontweight='bold')
    for c in rc:
        ax.scatter(c, 0, s=200, color='seagreen', zorder=3)
        ax.text(c, -0.18, f"r{c:+d}", ha='center', va='top', fontsize=8, color='seagreen', fontweight='bold')

    # Number line
    for x in range(int(xmin), int(xmax)+1):
        ax.axvline(x, color='lightgray', lw=0.5, zorder=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.45, 1.5)
    ax.set_title(f"λ = {tuple(partition)}", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['removable\n(R)', 'addable\n(A)'], fontsize=8)
    ax.set_xlabel("content  c = x − y", fontsize=8)

plt.tight_layout()
plt.savefig("data/plots/graph_a_matrix.png", dpi=130, bbox_inches="tight")
print("\nSaved plot → data/plots/graph_a_matrix.png")
