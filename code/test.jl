using Random
using Plots

# optional: choose backend
# gr()      # or plotlyjs(), pyplot(), etc.

# Make some example data
Random.seed!(1)
n = 20
A = round.(5 .* rand(n, n); digits=2)  # 0–5 with 2 decimals

# String labels for each cell
labels = string.(A)

# Build the heatmap
p = heatmap(
    0:n-1, 0:n-1, A;
    aspect_ratio=1,
    c=:viridis,          # colormap
    colorbar=false,
    xflip=false,
    yflip=true,          # to have (0,0) in top-left like your image
    xaxis=(nothing, 0:n-1),  # suppress label text, keep ticks
    yaxis=(nothing, 0:n-1),
    framestyle=:box,
    xgrid=true,
    ygrid=true,
    gridalpha=0.4,
)

# Add the numeric annotations
plot!(p, series_annotations=labels,
    annotationfontsize=6,  # adjust to taste
    linecolor=:transparent)  # hide lines from annotation series

display(p)