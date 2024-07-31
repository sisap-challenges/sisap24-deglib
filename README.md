# SISAP 2024 Challenge: working example on Julia 

This repository is a working example for the challenge <https://sisap-challenges.github.io/>, working with Python and GitHub Actions, as specified in Task's descriptions.


## Steps for running
It requires a working installation of python, and an installation of the git tools. You will need internet access for cloning and downloading datasets.

the following commands should be run
```bash
export DBSIZE=300K

mkdir data2024
cd data2024
curl -O https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=$DBSIZE.h5
curl -O http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5  # this url will be updated soon
curl -O http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=$DBSIZE--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 # this url will be updated soon
cd ..
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t auto task1.jl $DBSIZE
julia --project=. -t auto task3.jl $DBSIZE
julia --project=. eval.jl
```


## How to take this to create my own
You can fork this repository and polish it to create your solution or use it to see how input and output are made to adapt it to your similarity search pipeline. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap24-example-julia/actions>

 
