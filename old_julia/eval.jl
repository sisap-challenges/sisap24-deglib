using JLD2, SimilaritySearch, DataFrames, CSV, Glob, UnicodePlots

function evaluate_results(gfile, resultfiles, k)
    gold_knns = jldopen(f->f["knns"][1:k, :], gfile)
    res = DataFrame(size=[], algo=[], modelingtime=[], encdatabasetime=[], encqueriestime=[], buildtime=[], querytime=[], params=[], recall=[])
    for resfile in resultfiles
        @info resfile
        reg = jldopen(resfile) do f
            knns = f["knns"][1:k, :]
            recall = macrorecall(gold_knns, knns)
            push!(res, (f["size"], f["algo"],
                        get(f, "modelingtime", 0.0),
                        get(f, "encdatabasetime", 0.0),
                        get(f, "encqueriestime", 0.0),
                        f["buildtime"], f["querytime"], f["params"], recall))
        end

    end

    sort!(res, :recall)
    res
end

function print_results(f, D, gfile, files, task, dbsize)
    show(f, "text/plain", gfile => files); println(f)
    show(f, "text/plain", D); println(f)
    p = lineplot(D.recall; ylim=(0, 1), title=String(D.algo[1]), ylabel="recall", xlabel="$(D.params[1]) to $(D.params[end])")
    show(f, "text/plain", p); println(f)
end

if !isinteractive()
    goldsuffix = "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    k = 30
    if length(ARGS) == 0
        open("results-summary.txt", "w") do f
            for path in sort!(glob("results-task*/*"))
                task, dbsize = splitpath(path)
                lastpath = glob(joinpath(path, "*")) |> sort! |> last
                gfile = joinpath("data2024", "gold-standard-dbsize=$dbsize--$goldsuffix")
                files = glob(joinpath(lastpath, "*.h5"))
                length(files) == 0 && continue
                D = evaluate_results(gfile, files, k)
                println(f, "\n\n=== results for $dbsize $goldsuffix ===")
                print_results(stdout, D, gfile, files, task, dbsize)
                print_results(f, D, gfile, files, task, dbsize)
                CSV.write("$task-$dbsize.csv", D)
            end
        end
    else
        for path in ARGS
            task, dbsize, _ = splitpath(path)
            gfile = joinpath("data2024", "gold-standard-dbsize=$dbsize--$goldsuffix")
            files = glob(joinpath(path, "*.h5"))
            D = evaluate_results(gfile, files, k)
            println("\n\n=== results for $dbsize $goldsuffix ===")
            print_results(stdout, D, gfile, files, task, dbsize)
        end
    end
    
    nothing
end
