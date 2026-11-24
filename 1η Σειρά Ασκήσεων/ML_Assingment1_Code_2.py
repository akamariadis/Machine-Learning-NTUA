import numpy as np
import matplotlib.pyplot as plt

points = {
    'A': ((2, 0), +1),
    'B': ((2, 2), +1),
    'C': ((3, 1), +1),
    'D': ((0, 0), -1),
    'E': ((0, 2), -1),
    'F': ((-1, 1), -1),
}

support_vectors = {'A', 'B', 'D', 'E'}
pos_pts = np.array([coord for coord, y in points.values() if y == 1])
neg_pts = np.array([coord for coord, y in points.values() if y == -1])
sv_pts = np.array([points[k][0] for k in support_vectors])
fig, ax = plt.subplots(figsize=(8.5, 6))

ax.axvspan(0, 2, color='gray', alpha=0.12, label='Margin')
ax.axvline(x=1, color='black', linestyle='-', linewidth=2,
           label=r'Ευθεία απόφασης: $x_1 - 1 = 0$')
ax.axvline(x=2, color='tab:blue', linestyle='--', linewidth=1.5,
           label=r'Όριο $+1$: $x_1 - 1 = +1$')
ax.axvline(x=0, color='tab:red', linestyle='--', linewidth=1.5,
           label=r'Όριο $-1$: $x_1 - 1 = -1$')

ax.scatter(sv_pts[:, 0], sv_pts[:, 1], facecolors='none', edgecolors='black',
           s=280, linewidths=1.8, zorder=5, label='Support Vectors')

for name, ((x1, x2), y) in points.items():
    if y == 1:
        ax.annotate(f'{name}({x1},{x2})', (x1, x2),
                    xytext=(12, 8), textcoords='offset points',
                    ha='left', fontsize=10, fontweight='bold')
    else:
        ax.annotate(f'{name}({x1},{x2})', (x1, x2),
                    xytext=(-12, 8), textcoords='offset points',
                    ha='right', fontsize=10, fontweight='bold')

ax.annotate('', xy=(2, 1), xytext=(0, 1),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.8))
ax.text(1, 1.25, r'$\gamma = \frac{2}{\|w\|} = 2$', color='darkgreen',
        ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

ax.set_xlim(-2.3, 4.2)
ax.set_ylim(-0.8, 2.8)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.7)
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('Hard-Margin SVM — Ερώτημα (α)', fontsize=13)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()