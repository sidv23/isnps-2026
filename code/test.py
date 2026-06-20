
# %% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




# sample X from a two-moons distribution
np.random.seed(1)

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=20, noise=0.01, random_state=1)


np.random.seed(99) 
X, _ = make_moons(n_samples=20, noise=0.1, random_state=42)


# plot the points
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap='viridis')
plt.title('Sampled Points from Two-Moons Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.axis('equal')
plt.show()

# %% 

from scipy.spatial import distance_matrix
D = distance_matrix(X, X)

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=np.arange(1, D.shape[0]+1),
            yticklabels=np.arange(1, D.shape[0]+1))
plt.show()





# %% 

H = np.eye(len(D)) - np.ones(D.shape) / len(D)
B = -0.5 * H @ (D ** 2) @ H

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(B)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
sns.heatmap(B, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=np.arange(1, D.shape[0]+1),
            yticklabels=np.arange(1, D.shape[0]+1))
plt.show()




# %% 

eigenvalues, eigenvectors = np.linalg.eigh(B)
# sort in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

p = 2
L = np.diag(np.sqrt(eigenvalues[:p]))
V = eigenvectors[:, :p]


fig, ax = plt.subplots(figsize=(2,11))
sns.heatmap(V, square=False, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=[f"Dim {i+1}" for i in range(p)],
            yticklabels=np.arange(1, V.shape[0]+1))
ax.xaxis.tick_top()
plt.show()


# %% 

fig, ax = plt.subplots(figsize=(2,2))
sns.heatmap(L, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=[f"Dim {i+1}" for i in range(p)],
            yticklabels=[f"Dim {i+1}" for i in range(p)])
ax.xaxis.tick_top()
plt.show()




# %% 

X_embedded = V @ L
fig, ax = plt.subplots(figsize=(2,11))
sns.heatmap(X_embedded, square=False, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=[f"Dim {i+1}" for i in range(p)],
            yticklabels=np.arange(1, X_embedded.shape[0]+1))
ax.xaxis.tick_top()
plt.show()




# %%

import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def create_person_icon(size=(100, 100), color="skyblue"):
    """Generates a simple person icon (circle head + body)."""
    # Create a transparent image
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    w, h = size
    cx, cy = w / 2, h / 2
    
    # Head (Circle)
    head_radius = h * 0.15
    head_center = (cx, h * 0.2)
    draw.ellipse(
        [head_center[0] - head_radius, head_center[1] - head_radius,
         head_center[0] + head_radius, head_center[1] + head_radius],
        fill=color
    )
    
    # Body (Shoulders + torso)
    body_top = h * 0.4
    body_width = w * 0.6
    
    # Draw shoulder arc
    draw.chord(
        [cx - body_width/2, body_top, cx + body_width/2, body_top + body_width],
        start=180, end=0, fill=color
    )
    # Fill torso
    draw.rectangle(
        [cx - body_width/2, body_top + body_width/2, cx + body_width/2, h*0.85],
        fill=color
    )
    
    return img

# Create the icon
icon_img = create_person_icon(size=(64, 64), color="#1f77b4") # standard blue

# Create the graph
n = 20
G = nx.complete_graph(n)

# Layout
pos = nx.random_layout(G, seed=0)

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Draw edges
# Note: min_source_margin stops edges from overlapping the icons
nx.draw_networkx_edges(
    G,
    pos=pos,
    ax=ax,
    arrows=True,
    arrowstyle="-",
    alpha=0.2,
    edge_color='gray'
)

# Helpers to transform coordinates
tr_figure = ax.transData.transform
tr_axes = fig.transFigure.inverted().transform

# Calculate icon size relative to the plot
icon_scale = 0.04 
icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * icon_scale
icon_center = icon_size / 2.0

# Add icons as sub-axes
for n_node in G.nodes:
    xf, yf = tr_figure(pos[n_node])
    xa, ya = tr_axes((xf, yf))
    
    # Create a small axis for the icon centered at the node position
    a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
    a.imshow(icon_img)
    a.axis("off")

# %%

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def create_person_icon(size=(100, 100), color="#1f77b4"):
    """Generates a simple person icon (circle head + body)."""
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = w / 2, h / 2
    
    # Head
    head_radius = h * 0.15
    head_center = (cx, h * 0.2)
    draw.ellipse([head_center[0] - head_radius, head_center[1] - head_radius,
                  head_center[0] + head_radius, head_center[1] + head_radius], fill=color)
    
    # Body
    body_top = h * 0.4
    body_width = w * 0.6
    draw.chord([cx - body_width/2, body_top, cx + body_width/2, body_top + body_width],
               start=180, end=0, fill=color)
    draw.rectangle([cx - body_width/2, body_top + body_width/2, cx + body_width/2, h*0.85], fill=color)
    return img

# --- USER DATA SECTION ---
# Define your matrix X here (n_samples, 2)
# Example: X = np.array([[0.1, 0.5], [0.9, 0.2], ...])
# X = np.random.rand(20, 2) * 10 
# -------------------------

# Create the graph and layout
n = len(X)
G = nx.complete_graph(n)
pos = {i: X[i] for i in range(n)} # Map matrix rows to node positions

# Create custom icon
icon_img = create_person_icon(size=(64, 64))

# Setup plot
fig, ax = plt.subplots(figsize=(8, 8))

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax, arrows=True, arrowstyle="-",
    min_source_margin=20, min_target_margin=20,
    alpha=0.2, edge_color='gray'
)

# Helper functions for coordinate transformation
tr_figure = ax.transData.transform
tr_axes = fig.transFigure.inverted().transform

# Calculate icon size based on data scale
# 5% of the total X-axis range
icon_scale = 0.05 
icon_size_data = (ax.get_xlim()[1] - ax.get_xlim()[0]) * icon_scale
icon_size_display = icon_size_data # simplified for square aspect ratio

# Add icons
for n_node in G.nodes:
    xf, yf = tr_figure(pos[n_node])
    xa, ya = tr_axes((xf, yf))
    
    # Icon size in figure coordinates (0-1)
    # This ensures icons stay consistent relative to the plot size
    fig_icon_size = icon_scale 
    
    a = plt.axes([xa - fig_icon_size/2, ya - fig_icon_size/2, fig_icon_size, fig_icon_size])
    a.imshow(icon_img)
    a.axis("off")

# Add Labels
# We offset the text based on the data scale
y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06

for n_node, (x, y) in pos.items():
    ax.text(x, y - y_offset, f"{n_node+1}", ha='center', fontsize=10, zorder=10)

ax.axis('off')
plt.savefig('../images/plts/distgraph.png')
plt.show()
# %%

theta = np.pi / 5
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
X_rotated = X @ rotation_matrix.T

plt.figure(figsize=(6,6))
plt.scatter(X_rotated[:,0], X_rotated[:,1], s=100, cmap='viridis')
for i in range(len(X_rotated)):
    plt.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()
# %%
X
# %%

Delta = D
Xi = np.random.normal(0, 0.33, size=D.shape)
Xi = (Xi + Xi.T) / 2  # make symmetric
np.fill_diagonal(Xi, 0)  # zero diagonal
D_noisy = Delta + Xi

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D_noisy)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=np.arange(1, D_noisy.shape[0]+1),
            yticklabels=np.arange(1, D_noisy.shape[0]+1))
plt.show()
# %%

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D_noisy)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
sns.heatmap(D_noisy, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=np.arange(1, D_noisy.shape[0]+1),
            yticklabels=np.arange(1, D_noisy.shape[0]+1))
plt.show()
# %%

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D_noisy)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
# sns heatmap by absolute value but keep sign in annotation
cls = sns.diverging_palette(0, 0, as_cmap=True)
sns.heatmap(Xi, square=True, cmap=cls, annot=True, cbar=False, ax=ax, center=0.0,
            xticklabels=np.arange(1, D_noisy.shape[0]+1),
            yticklabels=np.arange(1, D_noisy.shape[0]+1))
plt.show()
# %% 

Xi2 = np.random.normal(0, 0.1, size=D.shape)
Xi2 = (Xi2 + Xi2.T) / 2  # make symmetric
np.fill_diagonal(Xi2, 0)  # zero diagonal
fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D_noisy)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
# sns heatmap by absolute value but keep sign in annotation
cls = sns.diverging_palette(0, 0, as_cmap=True)
sns.heatmap(Xi2, square=True, cmap=cls, annot=True, cbar=False, ax=ax, center=0.0,
            xticklabels=np.arange(1, D_noisy.shape[0]+1),
            yticklabels=np.arange(1, D_noisy.shape[0]+1))
plt.show()

# %%

def mds(D, p):
    H = np.eye(len(D)) - np.ones(D.shape) / len(D)
    B = -0.5 * H @ (D ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    L = np.diag(np.sqrt(eigenvalues[:p]))
    V = eigenvectors[:, :p]
    X_embedded = V @ L
    return X_embedded

X_noisy_embedded = mds(D_noisy, p=2)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_embedded[:,0], X_noisy_embedded[:,1], s=100, cmap='viridis')
for i in range(len(X_noisy_embedded)):
    ax.text(X_noisy_embedded[i,0], X_noisy_embedded[i,1]+0.05, str(i+1), ha='center', fontsize=10)
# ax.scatter(X_rotated[:,0], X_rotated[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
# for i in range(len(X_rotated)):
#     ax.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10, color='firebrick')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()

# %% 

D_noisy_2 = D + Xi2
X_noisy_embedded_2 = mds(D_noisy_2, p=2)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_embedded_2[:,0], X_noisy_embedded_2[:,1], s=100, cmap='viridis')
for i in range(len(X_noisy_embedded_2)):
    ax.text(X_noisy_embedded_2[i,0], X_noisy_embedded_2[i,1]+0.05, str(i+1), ha='center', fontsize=10)
# ax.scatter(X_rotated[:,0], X_rotated[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
# for i in range(len(X_rotated)):
#     ax.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10, color='firebrick')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()

# %%
def procrustes(X, Y):
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    X_centered = X - mu_X
    Y_centered = Y - mu_Y
    U, S, Vt = np.linalg.svd(Y_centered.T @ X_centered)
    R = U @ Vt
    Xr = (X_centered @ R) + mu_Y
    return Xr
X_noisy_aligned = procrustes(X_noisy_embedded, X_rotated)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_aligned[:,0], X_noisy_aligned[:,1], s=100, cmap='viridis')
for i in range(len(X_noisy_aligned)):
    ax.text(X_noisy_aligned[i,0], X_noisy_aligned[i,1]+0.05, str(i+1), ha='center', fontsize=10)
ax.scatter(X_rotated[:,0], X_rotated[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
for i in range(len(X_rotated)):
    ax.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10, color='firebrick')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()
# %% 

X_noisy_aligned_2 = procrustes(X_noisy_embedded_2, X_rotated)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_aligned_2[:,0], X_noisy_aligned_2[:,1], s=100, cmap='viridis')
for i in range(len(X_noisy_aligned_2)):
    ax.text(X_noisy_aligned_2[i,0], X_noisy_aligned_2[i,1]+0.05, str(i+1), ha='center', fontsize=10)
ax.scatter(X_rotated[:,0], X_rotated[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
for i in range(len(X_rotated)):
    ax.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10, color='firebrick')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()


# %% 

# plot X_noisy_aligned and X_noisy_aligned2 together
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_aligned[:,0], X_noisy_aligned[:,1], s=100, label='', alpha=0.7, cmap='viridis')
for i in range(len(X_noisy_aligned)):
    ax.text(X_noisy_aligned[i,0], X_noisy_aligned[i,1]+0.05, str(i+1), ha='center', fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
ax.legend()
plt.show()


# %% 

# plot X_noisy_aligned and X_noisy_aligned2 together
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_aligned[:,0], X_noisy_aligned[:,1], s=100, label='', alpha=0.7, cmap='viridis')
ax.scatter(X_noisy_aligned_2[:,0], X_noisy_aligned_2[:,1], s=100, label='', alpha=0.7, cmap='plasma')
for i in range(len(X_noisy_aligned)):
    ax.text(X_noisy_aligned[i,0], X_noisy_aligned[i,1]+0.05, str(i+1), ha='center', fontsize=10)
for i in range(len(X_noisy_aligned_2)):
    ax.text(X_noisy_aligned_2[i,0], X_noisy_aligned_2[i,1]-0.05, str(i+1), ha='center', fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
ax.legend()
plt.show()

# %%

X_noisy_aligned_rotated = X_noisy_aligned @ rotation_matrix.T
X_noisy_aligned_rotated
# %%
X_rotated
# %%

# ellipsoids around X_noisy_aligned
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_noisy_aligned[:,0], X_noisy_aligned[:,1], s=100, cmap='viridis')
for i in range(len(X_noisy_aligned)):
    ax.text(X_noisy_aligned[i,0], X_noisy_aligned[i,1]+0.05, str(i+1), ha='center', fontsize=10)
    ellipse = patches.Ellipse((X_noisy_aligned[i,0], X_noisy_aligned[i,1]), width=0.4, height=0.2,
                              angle=0, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(ellipse)
ax.scatter(X_rotated[:,0], X_rotated[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
for i in range(len(X_rotated)):
    ax.text(X_rotated[i,0], X_rotated[i,1]+0.05, str(i+1), ha='center', fontsize=10, color='firebrick')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()

# %%

X_cities = np.array([
    [  32.7157, -117.1611],
    [  47.6062, -122.3321],
    [  33.4484, -112.074 ],
    [  36.1699, -115.1398],
    [  45.676 , -111.0429],
    [  38.627 ,  -90.1994],
    [  41.8781,  -87.6298],
    [  41.4993,  -81.6944],
    [  40.4406,  -79.9959],
    [  40.7128,  -74.006 ],
    [  29.9511,  -90.0715],
    [  29.7604,  -95.3698],
    [  32.7767,  -96.797 ],
    [  35.4676,  -97.5164],
    [  37.6872,  -97.3301],
    [  38.0293,  -78.4767],
    [  42.3601,  -71.0589],
    [  33.749 ,  -84.388 ],
    [  28.5383,  -81.3792],
    [  35.687 , -105.9378],
    [  39.7392, -104.9903],
    [  40.7608, -111.891 ],
    [  37.7749, -122.4194],
    [  41.14  , -104.8202],
    [  41.2565,  -95.9345],
    [  40.8136,  -96.7026],
    [  44.9778,  -93.265 ],
    [  36.1627,  -86.7816],
    [  32.7765,  -79.9311],
    [  31.7619, -106.485 ],
    [  34.7465,  -92.2896]
])

X_cities = X_cities[:, ::-1]  # lat, lon to lon, lat
D_cities = distance_matrix(X_cities, X_cities)
Xi = np.random.normal(0, 2.5, size=D_cities.shape)
Xi = (Xi + Xi.T) / 2  # make symmetric
np.fill_diagonal(Xi, 0)  # zero diagonal
D_cities_noisy = D_cities + Xi

fig, ax = plt.subplots(figsize=(11,11))
# plot only upper triangle
mask = np.zeros_like(D_cities_noisy)
mask[np.tril_indices_from(mask)] = True
# sns.heatmap(D, square=True, cmap='viridis', annot=True, cbar=False, mask=mask, ax=ax)
ax.xaxis.tick_top()
# x and y axis from 1 to n
sns.heatmap(D_cities_noisy, square=True, cmap='viridis', annot=True, cbar=False, ax=ax,
            xticklabels=np.arange(1, D_cities_noisy.shape[0]+1),
            yticklabels=np.arange(1, D_cities_noisy.shape[0]+1))
plt.show()

# %%
X_cities_embedded = mds(D_cities_noisy, p=2)
X_cities_embedded_aligned = procrustes(X_cities_embedded, X_cities)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X_cities_embedded_aligned[:,0], X_cities_embedded_aligned[:,1], s=100, cmap='viridis')
ax.scatter(X_cities[:,0], X_cities[:,1], s=100, color='firebrick', alpha=1.0,marker='x')
for i in range(len(X_cities_embedded_aligned)):
    ax.text(X_cities_embedded_aligned[i,0], X_cities_embedded_aligned[i,1]+0.05, str(i+1), ha='center', fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axis('equal')
plt.show()

# %%
print(X_cities_embedded_aligned[:, ::-1])
# %%
