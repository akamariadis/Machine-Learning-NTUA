import numpy as np
import matplotlib.pyplot as plt

# β1

def relu(z):
    return np.maximum(0.0, z)

def T_exact(x):
    return np.where(x <= 0.5, 2 * x, 2 - 2 * x)

w = np.array([[1.0], [1.0]])
b = np.array([[0.0], [-0.5]])
a = np.array([[2.0, -4.0]])
c = 0.0

def mlp_T(x):
    x_row = np.atleast_2d(x)
    h = relu(w @ x_row + b)
    y = a @ h + c
    return y.ravel()

x_vals = np.linspace(0, 1, 10001)
max_error = np.max(np.abs(T_exact(x_vals) - mlp_T(x_vals)))

print("ΥΠΟΕΡΩΤΗΜΑ β1 -> MLP για T(x)")
print(f"Μέγιστο απόλυτο σφάλμα στο [0, 1]: {max_error:.2e}")

# β2

def T_k(x, k):
    y = x.copy()
    for _ in range(k):
        y = mlp_T(y)
    return y

def T2_analytical(x):
    return np.piecewise(
        x,
        [x <= 0.25, (x > 0.25) & (x <= 0.5), (x > 0.5) & (x <= 0.75), x > 0.75],
        [lambda x: 4*x, lambda x: 2 - 4*x, lambda x: 4*x - 2, lambda x: 4 - 4*x]
    )

y1 = T_k(x_vals, 1)
y2 = T_k(x_vals, 2)
y3 = T_k(x_vals, 3)

err_T2 = np.max(np.abs(y2 - T2_analytical(x_vals)))
print("\n ΥΠΟΕΡΩΤΗΜΑ β2 -> T_2(x) και T_3(x)")
print(f"Μέγιστο σφάλμα αναλυτικού τύπου T_2(x): {err_T2:.2e}")

slopes_T3 = np.diff(y3) / np.diff(x_vals)
slopes_rounded = np.round(slopes_T3, 2)
kinks_count = np.sum(np.diff(slopes_rounded) != 0)
segments_count = kinks_count + 1

print(f"Σημεία αλλαγής κλίσης της T_3(x): {kinks_count}")
print(f"Αριθμός ευθύγραμμων τμημάτων της T_3(x): {segments_count}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
titles = [r"$T_1(x) = T(x)$ (2 τμήματα)", r"$T_2(x)$ (4 τμήματα)", r"$T_3(x)$ (8 τμήματα)"]
ys = [y1, y2, y3]

for ax, y, title in zip(axes, ys, titles):
    ax.plot(x_vals, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.grid(True, linestyle="--", alpha=0.6)

axes[0].set_ylabel("y")
plt.tight_layout()
plt.show()

# β3

def deep_mlp_Tk(x, k):
    W_hidden = w @ a
    h = relu(w @ np.atleast_2d(x) + b)
    for _ in range(k - 1):
        h = relu(W_hidden @ h + b)

    y = a @ h + c
    return y.ravel()

err_deep = np.max(np.abs(deep_mlp_Tk(x_vals, 4) - T_k(x_vals, 4)))
print("\nΥΠΟΕΡΩΤΗΜΑ β3 -> Σύγκριση Βαθέος vs Ρηχού MLP")
print(f"Μέγιστο σφάλμα βαθέος MLP (k=4, 8 νευρώνες): {err_deep:.2e}\n")

k_values = np.arange(1, 11)
deep_neurons = 2 * k_values
shallow_lower_bound = 2 ** k_values - 1

print(f"{'k':>3} | {'Τμήματα (2^k)':>14} | {'Βαθύ MLP (2k)':>15} | {'Ρηχό MLP (>= 2^k - 1)':>22}")
print("-" * 64)
for k, d_n, s_n in zip(k_values, deep_neurons, shallow_lower_bound):
    print(f"{k:>3} | {2 ** k:>14} | {d_n:>15} | {s_n:>22}")

plt.figure(figsize=(7, 4.5))
plt.plot(k_values, deep_neurons, marker="o", label="Βαθύ MLP: $2k$ νευρώνες")
plt.plot(k_values, shallow_lower_bound, marker="s", label="Ρηχό MLP (κάτω φράγμα): $2^k - 1$ νευρώνες")
plt.yscale("log")
plt.xlabel("Βάθος $k$")
plt.ylabel("Συνολικός αριθμός κρυφών νευρώνων (log scale)")
plt.title("Ρόλος του βάθους στην αναπαραστατική ικανότητα των MLPs")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()