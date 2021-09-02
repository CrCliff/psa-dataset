# Out of the Park: Trading Card Grading Dataset and Models

Chris Clifford - 09/01/2021

Drexel CS 615

All code is published to https://github.com/CrCliff/psa-dataset.

## Installation

All dependencies are listed in requirements.txt. Please use a virtualenv as follows:

```
$ python -m venv env
$ source env/scripts/activate
$ pip install -r requirements
```

## Scraping

```
$ python main.py sc
```

## Preprocessing

```
$ python main.py pr
```

## Predicting

```
$ jupyter lab
```

Then open `eval.ipynb` and code away!
