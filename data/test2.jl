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

begin
    hsmembers = CSV.read(datadir("voting/HSall_members.csv"), DataFrame)
    transform!(hsmembers, :chamber => ByRow(x -> x) => :chamber1)
    hs_codes = hsmembers[!, [:icpsr, :party_code, :bioname, :chamber1]] |> unique

    hsvotes = CSV.read(datadir("voting/HSall_votes.csv"), DataFrame)
    transform!(hsvotes, :icpsr => ByRow(x -> convert(Int, x)) => :icpsr)

    hs = leftjoin(hsvotes, hs_codes, on=:icpsr)
    dropmissing!(hs, :chamber1)
    filter!(row -> row.chamber1 ∈ ["House", "Senate"], hs)
    select!(hs, Not(:chamber1))
    hs = hs |> unique
    DFs = groupby(hs, [:congress, :chamber])
end

congress_to_year(c::Int) = (1789 + 2c - 2, 1789 + 2c)
manchin_transform(df) = begin
    df[map(row -> occursin.("MANCHIN,", uppercase(row.bioname)), eachrow(df)), :icpsr] .= 90915
    df[map(row -> occursin.("MANCHIN,", uppercase(row.bioname)), eachrow(df)), :party_code] .= 328
    df
end

function makeDist(df, df_codes, F)
    D = zeros(fill(length(df_codes.icpsr), 2)...)
    @showprogress for (i, icp1) in enumerate(df_codes.icpsr), (j, icp2) in enumerate(df_codes.icpsr)
        v1 = df[df.icpsr.==icp1, [:icpsr, :rollnumber, :vote]]
        v2 = df[df.icpsr.==icp2, [:icpsr, :rollnumber, :vote]]
        common = innerjoin(v1, v2, on=:rollnumber, makeunique=true)
        n_common = nrow(common)
        if n_common == 0
            D[i, j] = 1.0
        else
            n_diff = reduce(+, abs.(skipmissing(common.vote .- common.vote_1)), init=0.0)
            D[i, j] = (n_diff / n_common)^2
        end
    end
    return D
end

function hsEmbed(DFs; cong=117, chamber="Senate", p=2, alpha=0.1, n_boot=5000, exclude=exclude, kwargs...)
    years = congress_to_year(cong)
    desc = "US Congress #$cong - $chamber ($(years[1]) - $(years[2]))"
    @info "Processing $desc"

    df = copy(DFs[(congress=cong, chamber=chamber)])
    transform!(df, :cast_code => ByRow(
        x -> x ∈ [1, 2, 3] ? 1 : x ∈ [4, 5, 6] ? -1 : x ∈ [7, 8, 9] ? 0 : missing
    ) => :vote)
    filter!(row -> row.icpsr ∉ exclude[cong], df)
    if cong > 116
        df = manchin_transform(df)
    end
    df_codes = dropmissing(unique(df[:, [:icpsr, :bioname, :party_code]]))
    D = makeDist(df, df_codes, (n_diff, n_common) -> (n_diff / n_common)^2)
    X = mds(D, p)
    mult = median(X[df_codes.party_code.==100, 1]) > 0 ? -1 : 1
    X[:, 1] = X[:, 1] .* mult
    Cα = BootConf(D, X, alpha=alpha; n_boot=n_boot)
    return D, X, Cα, df_codes, desc
end

begin
    notable_default = ["SANDERS", "COLLINS", "MANCHIN"]
    notable = [[] for _ in 1:120]
    exclude = [[99911, 99912, 99913, 99914] for _ in 1:120] # presidents

    exclude[103] = [15424]

    exclude[114] = []
    notable[114] = ["MURKOWSKI", "WARREN", "MCCASKILL", "SCHUMER", "DONNELLY", "KIRK", "GRAHAM", "RAND,", "VITTER", "CRUZ"]

    exclude[115] = [15429, 49700]
    notable[115] = ["KYL", "SESSIONS"]

    exclude[116] = [41904, 29909]
    notable[116] = ["MURKOWSKI", "WARREN", "MCCASKILL", "SCHUMER", "KIRK", "RAND,", "DOUG)", "CRUZ", "MENENDEZ", "ISAKSON", "LOEFFLER", "KELLY", "HARRIS", "THILLIS", "GRAHAM"]

    exclude[117] = [41904, 41701]
    notable[117] = ["LOEFFLER", "ISAKSON", "MCSALLY"]

    exclude[118] = [21937, 20104, 42306]
    # exclude[118] = []
    notable[118] = ["VANCE", "TILLIS", "WARREN", "MURKOWSKI", "PAUL", "FETTERMAN"]

    exclude[119] = [41102, 42304]
    notable[119] = ["VANCE", "TILLIS", "WARREN", "MURKOWSKI", "PAUL", "FETTERMAN"]
end

plotly()
gr()
begin
    cong = 116
    # cong = 103
    chamber, p = "Senate", 2
    D, X, Cα, df_codes, desc = hsEmbed(DFs,
        cong=cong, chamber=chamber, p=p, alpha=0.01, exclude=exclude, nboot=10000)


    Cα = [Ellipsoid(c.center, c.shape_matrix .* 2.0) for c in Cα]
    args, make_title = (;), true
    # args = (; args..., xlim=(-0.6, 0.6), ylim=(-1.4, 1.8))
    names = [notable_default..., notable[cong]...]
    title = make_title ? desc : ""
    cls = map(c -> c == 100 ? :dodgerblue : c == 200 ? :firebrick1 : :purple, df_codes.party_code)
    orient = map(c -> c == 100 ? :right : c == 200 ? :left : :right, df_codes.party_code)
    indx = map(c -> c == 100 ? 1 : c == 200 ? 2 : 1, df_codes.party_code)
    plt = plot(la=0, grid=false, title=title, size=(1000, 350), label="")
    plot!(plt, Cα, fa=0.1, c=permutedims(cls), la=0, label="")
    scatter!(plt, m2t(X), c=cls, msw=0.1, ms=4, ma=0.5, label="")
    for i in 1:1:size(X, 1)
        bioname = df_codes.bioname[i]
        # if any(occursin.(names, uppercase(bioname)))
        annotate!(plt, (X[i, :] .+ [0.001, 0.001])..., text(bioname, 5, :black, :left, rotation=30, color=:black), label="")
        # annotate!(plt, (X[i, :] .+ [0.001, 0.001])..., text(bioname, 10, :black, :left, rotation=30, color=:black), label="")
        # end
    end
    plot!(plt; label="", args...,
        ylim=(-0.75, 1.00),
        xlim=(-1.10, 0.90),
        # size=(1200, 300), grid=false)
        size=(900, 300), grid=false)
    plt
end

# filter Loeffler
filter(row -> occursin("ISAKSON", uppercase(row.bioname)), DFs[(congress=116, chamber=chamber)])

X



D, X, Cα, df_codes, desc = hsEmbed(DFs,
    cong=cong, chamber=chamber, p=p, alpha=0.01, exclude=exclude, nboot=10000)
Σα = [c.shape_matrix .* 2.0 for c in Cα]


# write D to CSV without row or column names
CSV.write("slides/senate_116_D.csv", DataFrame(D, :auto), writeheader=false)

# Write X to CSV without row or column names
CSV.write("slides/senate_116_X.csv", DataFrame(X, :auto), writeheader=false)

# Write df_codes to CSV
CSV.write("slides/senate_116_df_codes.csv", df_codes)

# Write Σα to JSON in the following format: a list of matrices
using JSON
open("slides/senate_116_Sigma.json", "w") do io
    JSON.print(io, Σα)
end