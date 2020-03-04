# SimMultiTrans
A Multimodes Transportation Simulator

## Prerequisite
### Download
```
git clone https://github.com/momodupi/SimMultiTrans.git
```

### Install the required packages

- Packages included in [Anaconda](https://www.anaconda.com/distribution/) (by default): [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/) 

- Other Packages: [Plotly](https://plot.ly/)

Install packages with [conda](https://docs.conda.io/en/latest/):
```
conda install numpy scipy matplotlib matplotlib-base mpl_sample_data pandas plotly
```

### Setup [Mapbox](https://www.mapbox.com/)
- [Copy `default public token`](https://account.mapbox.com/) (or create a new `token`)
- [Create a new `map style`](https://studio.mapbox.com/) and copy its URL
- Setup `token` and `map style`:
```
python setup.py
```

## Running
### Setup city parameters
- Modify `conf\city.json`
- Modify `conf\arr_rate.csv`

### Setup vehicle attributes
- Modify `conf\vehicle.json`

### Run a simple example
```
python main.py
```

### Review results
All generated results can be found in `result\`
