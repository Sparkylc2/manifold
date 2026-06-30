"""
Jansen leg position generator.

Single DOF: the crank angle theta. Every joint is solved by circle-circle
intersection (dyad solution) from the fixed-length bars. Outputs all joint
positions, validates every bar length, and prints initial positions for
seeding bodies + distance constraints.

Frame pivot O at the origin; crank center Oc at (a, l). Branch signs below
select the canonical (downward-hanging) Jansen assembly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Jansen "holy numbers" (mm)
a = 38.0   # x-offset  O -> Oc (crank center)
l = 7.8    # y-offset  O -> Oc
m = 15.0   # crank        Oc -> Pm
b = 41.5   # upper rocker  O  -> B2
c = 39.3   # lower rocker  O  -> B3
d = 40.1   # tri edge      O  -> D
e = 55.8   # tri edge      B2 -> D
f = 39.4   #               D  -> Jfg
g = 36.7   #               B3 -> Jfg
h = 65.7   #               Jfg-> foot
i = 49.0   #               B3 -> foot
j = 50.0   # upper coupler Pm -> B2
k = 61.9   # lower coupler Pm -> B3

O  = np.array([0.0, 0.0])
Oc = np.array([a, l])

# branch sign per solved joint (canonical assembly, validated by foot path)
# branch sign per solved joint. B2 (the j/b joint) takes the UPPER solution so
# the e-b-d triangle apex points up, matching the canonical Jansen diagram; this
# is the unique non-self-intersecting assembly over a full revolution.
BRANCH = {'B2': +1, 'D': +1, 'B3': -1, 'Jfg': -1, 'P': -1}

# bar -> (endpoint, endpoint, rest length) : maps 1:1 to distance constraints
BARS = [
    ('m', 'Oc', 'Pm', m), ('b', 'O', 'B2', b), ('j', 'Pm', 'B2', j),
    ('c', 'O', 'B3', c),  ('k', 'Pm', 'B3', k), ('d', 'O', 'D', d),
    ('e', 'B2', 'D', e),  ('f', 'D', 'Jfg', f), ('g', 'B3', 'Jfg', g),
    ('i', 'B3', 'P', i),  ('h', 'Jfg', 'P', h),
]

def circ(p1, r1, p2, r2, s):
    """Intersection of circle(p1,r1) & circle(p2,r2); s = +/-1 picks the branch."""
    dv = p2 - p1
    dist = np.hypot(*dv)
    aa = (r1**2 - r2**2 + dist**2) / (2 * dist)
    hh = np.sqrt(max(r1**2 - aa**2, 0.0))
    perp = np.array([-dv[1], dv[0]]) / dist
    return p1 + aa * dv / dist + s * hh * perp

def solve(theta):
    """All joint positions for crank angle theta (rad)."""
    Pm  = Oc + m * np.array([np.cos(theta), np.sin(theta)])
    B2  = circ(O,   b, Pm, j, BRANCH['B2'])
    D   = circ(O,   d, B2, e, BRANCH['D'])
    B3  = circ(O,   c, Pm, k, BRANCH['B3'])
    Jfg = circ(D,   f, B3, g, BRANCH['Jfg'])
    P   = circ(Jfg, h, B3, i, BRANCH['P'])
    return dict(O=O, Oc=Oc, Pm=Pm, B2=B2, D=D, B3=B3, Jfg=Jfg, P=P)

def max_bar_error(P):
    return max(abs(np.hypot(*(P[p] - P[q])) - L) for _, p, q, L in BARS)

if __name__ == "__main__":
    thetas = np.deg2rad(np.arange(0, 360, 1))
    worst = max(max_bar_error(solve(t)) for t in thetas)
    print(f"max bar-length error over full rotation: {worst:.2e} mm\n")

    P0 = solve(0.0)
    print("initial joint positions (crank angle = 0):")
    for nm in ['O', 'Oc', 'Pm', 'B2', 'D', 'B3', 'Jfg', 'P']:
        print(f"  {nm:4s} = ({P0[nm][0]:9.4f}, {P0[nm][1]:9.4f})")

    print("\nbars (body_a, body_b, rest_length) -> distance constraints:")
    for nm, p, q, L in BARS:
        print(f"  {nm}: {p:>3s} -- {q:<3s}  L = {L}")

    foot = np.array([solve(t)['P'] for t in thetas])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(foot[:, 0], foot[:, 1], 'r-')
    ax[0].set_title("foot path"); ax[0].axis('equal'); ax[0].grid(True)
    for nm, p, q, L in BARS:
        ax[1].plot([P0[p][0], P0[q][0]], [P0[p][1], P0[q][1]], 'b-', lw=1.6)
    for nm in P0:
        ax[1].plot(*P0[nm], 'ko', ms=3); ax[1].annotate(nm, P0[nm])
    ax[1].plot(foot[:, 0], foot[:, 1], 'r-', lw=.7)
    ax[1].set_title("linkage @ theta=0"); ax[1].axis('equal'); ax[1].grid(True)
    plt.tight_layout(); plt.savefig("jansen_linkage.png", dpi=120)
    print("\nsaved jansen_linkage.png")
