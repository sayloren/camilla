# camilla

[![Build
Status](https://travis-ci.org/sayloren/camilla.svg?branch=master)](https://travis-ci.org/sayloren/camilla)

[Travis Build Results](https://travis-ci.org/sayloren/camilla)

:see_no_evil: :hear_no_evil: :speak_no_evil:

## UCSF BMI 203 Algorithms Homework 2018

#### Set up

To use the package, first make a new conda environment and activate it

```
conda create -n final python=3
source activate final
```

to install all the requirements run

```
conda install --yes --file requirements.txt
```

##### To run neural net
```
cd <~/camilla/nn/>
python -m 
```

##### To run sort tests
```
python -m pytest -v test/run_tests.py
```

##### Graphs for Seq Learning Rate x other params
![a](/images/Learningrate-error-seq.png)

##### Graphs for Seq Weight/Bias relations over Epochs
![a](/images/Weights-biases-seq.png)
