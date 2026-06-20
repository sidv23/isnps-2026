using DrWatson
@quickactivate
cd(projectdir());

begin
    using Pipe
    using LinearAlgebra, Random
    using KernelFunctions, Distances
    using Distributions, Statistics, VectorizedStatistics
    using Plots, Plots.Measures
    using ProgressMeter
    using Distributions
    using LaTeXStrings
    using LazySets
    using StatsPlots
    using StatsBase: sample, Weights
    using SpecialFunctions
    using JLD2
    using JSON3
    using CSV, DataFrames
end

for filename in readdir(srcdir())
    if endswith(filename, ".jl")
        includet(joinpath(srcdir(), filename))
    end
end


X = [
    0.83688247 0.79822634;
    1.27989123 0.82798965;
    1.22378615 0.26962226;
    0.35235277 0.86256038;
    -0.98413894 -0.24831009;
    -0.73453377 0.78651725;
    -0.06605116 0.14359662;
    -0.13590003 1.05388491;
    1.46699321 1.25700387;
    1.41382041 0.47855507;
    0.60349577 0.18143694;
    0.19006111 0.97903900;
    -0.88150206 -0.69101033;
    1.19840654 1.67713394;
    -0.33033692 0.80504970;
    0.15833022 -0.19049424;
    -1.00264876 0.10613539;
    -0.12559967 0.29976632;
    0.99766274 0.12874113;
    -0.91400957 0.48787988
]


Y = [
    0.95223147 0.81585689;
    1.31687880 0.87814367;
    1.18201115 0.27457908;
    0.44913664 0.71188480;
    -1.08430524 -0.11424156;
    -0.68926032 0.69689866;
    0.08289510 -0.03393953;
    -0.11822986 1.11809578;
    1.39121717 1.25195167;
    1.26939607 0.38280606;
    0.56999665 0.17149141;
    0.07260755 0.89062749;
    -0.80760432 -0.64391158;
    1.05462093 1.89937399;
    -0.17985529 0.75218213;
    -0.08794704 -0.30519752;
    -1.12197952 0.04158958;
    0.00385477 0.52984391;
    1.05157324 0.13012897;
    -0.76027620 0.56516006
]

plt = plot(size=(400, 400), legend=false, grid=false)
scatter!(tup(X), c=:red, m=:X, msw=2.0)
scatter!(tup(Y), c=:blue, m=:circle, msw=0.0, ms=3.0)


rot_mat = θ -> [cos(θ) -sin(θ); sin(θ) cos(θ)]
ITER = 50
Rs = rand(n, ITER) .+ 0.1
n, p = size(X)
thetas = range(0, 2pi, length=ITER)
@gif for i in 1:length(thetas)
    Random.seed!(42)
    T = rand(Uniform(0, 2pi), n)
    eigens = [Symmetric(rot_mat(T[i]) * Diagonal(rand(2) .* 0.1) * rot_mat(T[i])') for i in 1:n]
    R = sin(thetas[i] / 2) + 0.2
    θ = thetas[i]
    # eigens = [r * Symmetric(rot_mat(θ) * e * rot_mat(θ)') for (r, e) in zip(R, eigens)]
    eigens = [R .* Symmetric(rot_mat(θ) * e * rot_mat(θ)') for e in eigens]
    Es = [Ellipsoid(x, e) for (x, e) in zip(eachrow(Y), eigens)]

    plt = plot(size=(400, 400), legend=false, grid=false, xlim=(-1.5, 2.0), ylim=(-1, 2.2), aspect_ratio=:equal, background_color=:white)
    plot!(Es, c=:gray, alpha=0.2, msw=0.0, lw=0.0)
    scatter!(tup(X), c=:red, m=:X, msw=2.0)
    scatter!(tup(Y), c=:blue, m=:circle, msw=0.0, ms=3.0)
    plot!(grid=false)
end fps=7