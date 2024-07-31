using SimilaritySearch, SurrogatedDistanceModels, HDF5, JLD2, CSV, Glob, LinearAlgebra, Dates

include("common.jl")

# This file is based in the julia's example of the 2023 edition


function run_search_task3(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::String)
    resfile_ = replace(resfile_, ".h5" => "")
    step = 1.05f0
    delta = idx.search_algo.Δ / step^3
    params = meta["params"]

    # produces result files for different search hyperparameters
    while delta < 2f0
        idx.search_algo.Δ = delta
        dt = "delta=$(round(delta; digits=3))"
        resfile = "$resfile_-$dt.h5"
        @info "searching $resfile"
        meta["params"] = "$params $dt"
        querytime = @elapsed knns, dists = searchbatch(idx, queries, k)
        meta["querytime"] = querytime
        save_results(knns, dists, meta, resfile)
        delta *= step
    end
end

function task3(;
        dbsize,
        dfile="data2024/laion2B-en-clip768v2-n=$dbsize.h5",
        qfile="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        k=30,
        outdir="results-task3/$dbsize/$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))"
    )

    mkpath(outdir) 
    dist = NormalizedCosineDistance()  # 1 - dot(·, ·)
    nbits = 8 * 4 * 128  # memory eq to 128 fp32 
    #model, dist_proj, nick = create_rp_model(dist, dfile; nbits)
    modelingtime = @elapsed model, dist_proj, nick = create_pca_model(dist, dfile; nbits)
    encdatabasetime = @elapsed db = predict_h5(model, dfile; nbits)
    encqueriestime = @elapsed queries = predict_h5(model, qfile; nbits)

    # loading or computing knns
    @info "indexing, this can take a while!"
    G, meta = build_searchgraph(dist_proj, db)
    meta["size"] = dbsize
    meta["modelingtime"] = modelingtime
    meta["encdatabasetime"] = encdatabasetime
    meta["encqueriestime"] = encqueriestime
    meta["params"] = "$(meta["params"]) $nick"
    resfile = joinpath(outdir, "searchgraph-$nick-k=$k")
    run_search_task3(G, queries, k, meta, resfile)
end

# functions for each database; these should have all required hyperparameters

if !isinteractive()
    if length(ARGS) == 0 || any(dbsize -> dbsize ∉ ("300K", "10M", "100M"), ARGS)
        throw(ArgumentError("this script must be called with a list of the following arguments: 300K, 10M or 100M"))
    end

    for dbsize in ARGS
        task3(; dbsize)
    end
end
