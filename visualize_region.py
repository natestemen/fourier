import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Region: pi/4 >= x >= y >= |z|
# i.e. |z| <= y <= x <= pi/4

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# --- Filled volume via scatter of valid points ---
N = 40
xs = np.linspace(0, np.pi / 4, N)
ys = np.linspace(0, np.pi / 4, N)
zs = np.linspace(-np.pi / 4, np.pi / 4, N)

X, Y, Z = np.meshgrid(xs, ys, zs)
mask = (X <= np.pi / 4) & (Y <= X) & (np.abs(Z) <= Y)

ax.scatter(
    X[mask], Y[mask], Z[mask],
    c=X[mask], cmap='plasma', alpha=0.04, s=6, linewidths=0
)

# --- Boundary surfaces ---
alpha_surf = 0.25

# Surface 1: x = pi/4 (cap), for 0 <= y <= pi/4, |z| <= y
y1 = np.linspace(0, np.pi / 4, 50)
z1 = np.linspace(-np.pi / 4, np.pi / 4, 50)
Y1, Z1 = np.meshgrid(y1, z1)
X1 = np.full_like(Y1, np.pi / 4)
mask1 = np.abs(Z1) <= Y1
X1[~mask1] = np.nan
ax.plot_surface(X1, Y1, Z1, color='royalblue', alpha=alpha_surf)

# Surface 2: y = x (wedge face), for 0 <= x <= pi/4, |z| <= x
x2 = np.linspace(0, np.pi / 4, 50)
z2 = np.linspace(-np.pi / 4, np.pi / 4, 50)
X2, Z2 = np.meshgrid(x2, z2)
Y2 = X2.copy()
mask2 = np.abs(Z2) <= Y2
X2[~mask2] = np.nan; Y2[~mask2] = np.nan; Z2[~mask2] = np.nan
ax.plot_surface(X2, Y2, Z2, color='tomato', alpha=alpha_surf)

# Surface 3: z = +y (upper cone face), for 0 <= y <= x <= pi/4
x3 = np.linspace(0, np.pi / 4, 50)
y3 = np.linspace(0, np.pi / 4, 50)
X3, Y3 = np.meshgrid(x3, y3)
Z3 = Y3.copy()
mask3 = Y3 <= X3
X3[~mask3] = np.nan; Y3[~mask3] = np.nan; Z3[~mask3] = np.nan
ax.plot_surface(X3, Y3, Z3, color='mediumseagreen', alpha=alpha_surf)

# Surface 4: z = -y (lower cone face)
Z3n = -Y3.copy()
Z3n[~mask3] = np.nan
ax.plot_surface(X3, Y3, Z3n, color='mediumseagreen', alpha=alpha_surf)

# --- Edges / ridge lines ---
t = np.linspace(0, np.pi / 4, 200)

# Edge: x = pi/4, y = pi/4, z in [-pi/4, pi/4]  (not part of region — just vertex)
# Edge: z=0, y=0, x from 0 to pi/4 (spine)
ax.plot([0, np.pi/4], [0, 0], [0, 0], 'k-', lw=1.5, alpha=0.7)
# Edge: z=y=x (upper ridge)
ax.plot(t, t, t, 'k-', lw=1.5, alpha=0.7)
# Edge: z=-y=−x (lower ridge)
ax.plot(t, t, -t, 'k-', lw=1.5, alpha=0.7)
# Edge: x=pi/4, y from 0 to pi/4, z=y
ax.plot([np.pi/4]*len(t), t, t, 'k-', lw=1.2, alpha=0.5)
# Edge: x=pi/4, y from 0 to pi/4, z=-y
ax.plot([np.pi/4]*len(t), t, -t, 'k-', lw=1.2, alpha=0.5)
# Edge: x=y, z=0, x from 0 to pi/4
ax.plot(t, t, np.zeros_like(t), 'k--', lw=1, alpha=0.4)

# --- Labels and formatting ---
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_zlabel('z', fontsize=13)
ax.set_title(r'Region: $\frac{\pi}{4} \geq x \geq y \geq |z|$', fontsize=15)

ax.set_xticks([0, np.pi/4]); ax.set_xticklabels(['0', 'π/4'])
ax.set_yticks([0, np.pi/4]); ax.set_yticklabels(['0', 'π/4'])
ax.set_zticks([-np.pi/4, 0, np.pi/4]); ax.set_zticklabels(['-π/4', '0', 'π/4'])

ax.set_xlim(0, np.pi/4)
ax.set_ylim(0, np.pi/4)
ax.set_zlim(-np.pi/4, np.pi/4)
ax.set_box_aspect([1, 1, 2])  # x,y span pi/4; z spans pi/2 — so z gets 2x

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='royalblue',    alpha=0.5, label=r'$x = \pi/4$'),
    Patch(facecolor='tomato',       alpha=0.5, label=r'$y = x$'),
    Patch(facecolor='mediumseagreen', alpha=0.5, label=r'$|z| = y$'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

ax.view_init(elev=20, azim=-50)
plt.tight_layout()
plt.savefig('region_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to region_plot.png")
