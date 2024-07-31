using SimilaritySearch, SurrogatedDistanceModels, HDF5, JLD2, CSV, Glob, LinearAlgebra, Dates

include("common.jl")

# This file is based in the julia's example of the 2023 edition

function postprocessing(idx, queries, knns, dists, k, dfile, qfile)
    m, n = size(knns)
    dist = NormalizedCosineDistance()
    knns_ = zeros(Int32, k, n)
    dists_ = zeros(Float32, k, n)
    ctx = getcontext(idx)
     
    h5open(dfile) do fX
        h5open(qfile) do fQ
            X, Q = fX["emb"], fQ["emb"]
            L = Threads.SpinLock() 

            Threads.@threads :static for i in eachindex(queries)
                res = getknnresult(k, ctx)
                K = sort!(knns[:, i])
                q = lock(L) do
                    Q[:, i]
                end

                for j in 1:m
                    #objID = Int(knns[j, i])
                    objID = K[j]
                    objID == 0 && continue
                    v = lock(L) do
                        X[:, objID]
                    end
                    d = evaluate(dist, q, v)
                    push_item!(res, objID, d)
                end

                knns_[:, i] .= IdView(res)
                dists_[:, i] .= DistView(res)
            end
        end
    end
    
    knns_, dists_
end

function run_search_task2(idx::SearchGraph, queries::AbstractDatabase, k::Integer, kspanlist::Vector, meta, resfile_::String, dfile, qfile)
    resfile_ = replace(resfile_, ".h5" => "")
    params = meta["params"]

    # produces result files for different search hyperparametersfor k2 delta < 2f0
    for kspan in kspanlist
        dt = "kspan=$kspan"
        resfile = "$resfile_-$dt.h5"
        @info "searching $resfile"
        meta["params"] = "$params $dt"
        querytime = @elapsed knns, dists = searchbatch(idx, queries, kspan)
        if k != kspan
            querytime += @elapsed knns, dists = postprocessing(idx, queries, knns, dists, k, dfile, qfile)
        end
        meta["querytime"] = querytime
        save_results(knns, dists, meta, resfile)
    end
end

function task2(;
        dbsize,
        dfile="data2024/laion2B-en-clip768v2-n=$dbsize.h5",
        qfile="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        k=30,
        kspanlist = [k, 3k], # More realistic kspanlist: [k, 3k, 10k, 30k],
        outdir="results-task2/$dbsize/$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))"
    )

    mkpath(outdir) 
    dist = NormalizedCosineDistance()  # 1 - dot(·, ·)
    nbits = 8 * 4 * 96 # same than 96 FP32 
    modelingtime = @elapsed model, dist_proj, nick = create_pca_model(dist, dfile; nbits)
    encdatabasetime = @elapsed db = predict_h5(model, dfile; nbits)
    encqueriestime = @elapsed queries = predict_h5(model, qfile; nbits)

    # loading or computing knns
    @info "indexing, this can take a while!"
    G, meta = build_searchgraph(dist_proj, db)
    meta["modelingtime"] = modelingtime
    meta["encdatabasetime"] = encdatabasetime
    meta["encqueriestime"] = encqueriestime
    meta["size"] = dbsize
    meta["params"] = "$(meta["params"]) $nick-$nbits"
    resfile = joinpath(outdir, "searchgraph-$nick-k=$k")
    run_search_task2(G, queries, k, kspanlist, meta, resfile, dfile, qfile)
end

# functions for each database; these should have all required hyperparameters
if !isinteractive()
    if length(ARGS) == 0 || any(dbsize -> dbsize ∉ ("300K", "10M", "100M"), ARGS)
        throw(ArgumentError("this script must be called with a list of the following arguments: 300K, 10M or 100M"))
    end

    for dbsize in ARGS
        task2(; dbsize)
    end
end
