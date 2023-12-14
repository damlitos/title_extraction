
# Development
Update both the requirements.txt  when changing any packages or package versions.
If you fail to do this the deployments might fail.
 
# Installation
## PIP
To run in your local python environment
```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install .
titleExtractor
```

## Conda
To run the code locally using conda enviroment
```
conda env create -f ./requirements.txt
conda activate titleExtractor
pip install .
titleExtractor
```

