using HDF5, FileIO, Glob, SimilaritySearch, DataFrames, CSV, OhMyREPL
D = let L = glob("deglib_index=*h5") 
  gold = load("/home/sisap23evaluation/data2024/gold-standard-dbsize=100M--private-queries-2024-laion2B-en-clip768v2-n=10k.h5", "knns")
  D = DataFrame(eps=[], algo=[], params=[], buildtime=[], querytime=[], recall=[])
  for filename in L 
      knns, algo, params, buildtime, querytime = h5open(filename) do f
          f["knns"][], read_attribute(f, "algo"), read_attribute(f, "params"), read_attribute(f, "buildtime"), read_attribute(f, "querytime")
      end

      recall = macrorecall(view(gold, 1:30, :), knns)
      eps = match(r"eps=(\d+.\d+)", filename).captures[1]
      push!(D, (; eps, algo, params, buildtime, querytime, recall))
  end
  D
end
CSV.write("../../../task1-results.csv", D)
