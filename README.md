# SISAP 2024 Challenge: working example on Julia 

This repository is a working example for the challenge <https://sisap-challenges.github.io/>, working with Python and GitHub Actions, as specified in Task's descriptions.


## Steps for running
It requires a working installation of python, and an installation of the git tools. You will need internet access for cloning and downloading datasets.

the following commands should be run
```bash
export DBSIZE=300K

mkdir data
cd data
curl -O https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=$DBSIZE.h5
curl -O http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5  
curl -O http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=$DBSIZE--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 
cd ..

# for task 1
python run_task.py --dbsize $DBSIZE --compression 512 --query-file "data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"

# for task 3
python run_task.py --dbsize $DBSIZE --compression 64 --query-file "data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"

# evaluate
python calc_recall.py --dbsize DBSIZE --ground-truth data/gold-standard-dbsize\=$DBSIZE--public-queries-2024-laion2B-en-clip768v2-n\=10k.h5 
```

The res.csv file is in the "results" directory.