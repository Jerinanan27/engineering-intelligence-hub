"""One-off generator for a sample architecture diagram (so the repo ships with
a real image to ingest). Run: python scripts/make_sample_diagram.py
Requires matplotlib (dev-only; NOT a runtime dependency of the app)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                linewidth=1.5, edgecolor="#1f3a5f", facecolor=color))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1f3a5f")

def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.4, color="#33506b"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                fontsize=8, color="#33506b", style="italic")

box(0.4, 2.6, 1.9, 1.0, "Client", "#eaf2fb")
box(3.0, 2.6, 2.0, 1.0, "API Gateway", "#d6e6f7")
box(5.9, 4.0, 2.0, 1.0, "Auth Service\n(auth-svc)", "#cfe3f5")
box(5.9, 1.2, 2.0, 1.0, "Vault\nsecret/auth/jwt", "#f7ead6")
box(8.3, 4.0, 1.4, 1.0, "JWKS\n/.well-known", "#e6f5e0")

arrow(2.3, 3.1, 3.0, 3.1, "Bearer JWT")
arrow(5.0, 3.3, 5.9, 4.3, "verify token")
arrow(5.9, 1.8, 5.0, 3.0, "X-User-Claims")
arrow(7.9, 4.5, 8.3, 4.5, "fetch keys")
arrow(6.9, 4.0, 6.9, 2.2, "RS256 key")

ax.text(5, 5.6, "Authentication Flow (RS256 JWT)", ha="center",
        fontsize=13, fontweight="bold", color="#1f3a5f")

plt.tight_layout()
fig.savefig("data/sample/diagrams/auth-architecture.png", dpi=130,
            bbox_inches="tight", facecolor="white")
print("wrote data/sample/diagrams/auth-architecture.png")
