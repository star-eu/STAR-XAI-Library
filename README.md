# STAR XAI LIBRARY : A novel framework for eXplainable Artificial Intelligence (XAI) methods and techniques for industrial data

## Description
This repository is based  on the work done within the STAR H2020 project and incorporates our paper with title: 
[XAI enhancing cyber defence against adversarial
attacks in industrial applications"](https://rdcu.be/b3Vh2) [1], and it can be used on Image data and ML techniques. 
Feel free to reuse, modify and extend this repository.

## STAR-XAI framework
Multi-nilm is a novel framework for efficient explanation of ML outputs applied on manufacturing data. 
It has four main categories:
- It provides a set of XAI models towards explaining Image data.
- It provides a set of XAI models towards explaining timeseries data.
- It provides a set of XAI models towards explaining text data.
- It provides a set of XAI models towards explaining tabular data.

[//]: # (## Examples)

[//]: # (Examples of experiments can be found under the directory _experiments_. )

[//]: # (The module [experiments.py]&#40;experiments/experiments.py&#41; defines three types of experiments &#40;_GenericExperiment, ModelSelectionExperiment_ )

[//]: # (and _REDDModelSelectionExperiment_&#41;. You can also create your own )

[//]: # (experiment by extending the abstract class _nilmlab.lab.Experiment_.)

[//]: # (After defining an experiment it requires only a few lines of code to setup and configure it. )

[//]: # (All files with names _run*.py_ are specific implementations that can be used as a reference.)

[//]: # (In order to run any of them it is as simple as: )

[//]: # (```python)

[//]: # (python -m experiments.run_generic_experiment)

[//]: # (```)

[//]: # (The results are saved under the directory _results_ as a csv file containing information about the )

[//]: # (setup, the source of the data, the parameters, the classification models, the performance and others.)

## Data

Currently only **Image** data are supported.


## Project structure
A detailed structure of the project is presented below. The key points are:
   - 📂 __data\_exploration__: Contains helpful notebooks e.g. explanation of each xai model.
   - 📂 __datasources__: Includes modules related to data e.g. loading, processing labels and others. 
   - 📂 __experiments__: Defines some experiments such as model selection and has examples on how to run the 
   defined experiments. 
   - 📂 __xai_models__: This is the main code which encapsulates all the logic of the proposed framework 
   and implements various XAI techniques.
   - 📂 __pretrained\_models__: Any pretrained models that are used for prediction.
   - 📂 __results__: Results of the experiments will be saved in this directory.
   - 📂 __utils__: Various tools that have been developed to support the implementation of the various algorithms.


- 📂 __multi\-nilm__
   - 📄 [LICENSE](LICENSE)
   - 📄 [README.md](README.md)
   - 📄 [requirements.txt](requirements.txt)
   - 📂 __data\_exploration__: 
     - 📄 [\_\_init\_\_.py](data_exploration/__init__.py)
     - 📄 [star\_data\_shaver\-all.ipynb](data_exploration/star_data_shaver-all.ipynb)
   - 📂 __datasources__
     - 📄 [\_\_init\_\_.py](datasources/__init__.py)
   - 📂 __experiments__
   - 📂 __pretrained\_models__
     - 📄 [clf\-v1.pkl](pretrained_models/clf-v1.pkl)
     - 📄 [signal2vec\-v1.csv](pretrained_models/signal2vec-v1.csv)
   - 📂 __results__
     - 📄 [\_\_init\_\_.py](results/__init__.py)
   - 📂 __utils__
     - 📄 [\_\_init\_\_.py](utils/__init__.py)
     - 📄 [helper\_functions.py](utils/helper_functions.py)
     - 📄 [logger.py](utils/logger.py)
   - 📂 __xai\_models__
     - 📄 [\_\_init\_\_.py](xai_models/__init__.py)
     - 📄 [BaseImageExplainer.py](xai_models/BaseImageExplainer.py)
     - 📄 [imageexplainers.py](xai_models/imageexplainers.py)


## Dependencies

The code has been developed using python3.6 and the dependencies can be found in [requirements.txt](requirements.txt).
- numpy~=1.21.5
- opencv-python~=4.5.5.64
- tensorflow~=2.9.1
- loguru~=0.4.1
- scikit-image~=0.19.2




## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## References

[//]: # (1. Nalmpantis, C., Vrakas, D. On time series representations for multi-label NILM. Neural Comput & Applic &#40;2020&#41;. https://doi.org/10.1007/s00521-020-04916-5)
[//]: # (2. Nalmpantis, C., & Vrakas, D. &#40;2019, May&#41;. Signal2Vec: Time Series Embedding Representation. In International Conference on Engineering Applications of Neural Networks &#40;pp. 80-90&#41;. Springer, Cham. https://doi.org/10.1007/978-3-030-20257-6_7)

