# SISAP 2024 Indexing Challenge

This branch contains the code for our submission (Team "HTW") to the SISAP 2024 Indexing Challenge.

**Members:**

- Nico Hezel, HTW Berlin, Germany
- Kai Barthel, HTW Berlin, Germany
- Konstantin Schall, HTW Berlin, Germany
- Klaus Jung, HTW Berlin, Germany
- Bruno Schilling, HTW Berlin, Germany

## Setup
It requires a working installation of python, and an installation of the git tools. You will need internet access for cloning and downloading datasets. 
See also `.github/workflows/ci.yml` file for a running configuration. Note the different parameters for 300K and 100M datasets when running the experiments.

### Install requires packages with pip 
```bash
python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deglib==0.1.51
python3 -m pip install h5py tensorflow
```

### Run Experiments
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