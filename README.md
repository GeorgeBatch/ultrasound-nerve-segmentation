# Kaggle Ultrasound Nerve Segmentation Competition (2016)<br>as an Undergraduate Research Project.

## Warning!!!

The code was run on Kaggle machines in July 2019 - October 2019 using Python 3 interpreter. It might stop working in the future due to the changes to Kaggle virtual environment and the updates of Python libraries used in this code. Please check [Kaggle documentation] and [kaggle/python docker image] for more information.

[Kaggle documentation]: https://www.kaggle.com/docs/kernels#the-kernels-environment
[kaggle/python docker image]: https://github.com/kaggle/docker-python

## About the Project

### Project Details:
* Title: Machine learning in the service of surgeons
* Author: George Batchkala, g.batchkala@warwick.ac.uk
* Supervisor: Dr Sigurd Assing, s.assing@warwick.ac.uk
* Institution: University of Warwick
* Department: Statistics
* Project funding: Undergraduate Research Support Scheme at the University of Warwick
* Project's official dates: July 1st 2019 - August 29th 2019
* Project's real dates: July 1st 2019 - August 29th 2019, October 2019
* Data Set: Kaggle "Ultrasound Nerve Segmentation" (2016) <br>https://www.kaggle.com/c/ultrasound-nerve-segmentation/overview
* Project's GitHub repository: https://github.com/GeorgeBatch/ultrasound-nerve-segmentation

### Motivation
In summer 2019, I was conducting an undergraduate research project within the Statistics Department of the University of Warwick. Together with my supervisor, both being interested in machine learning, we chose to work on an old Kaggle competition. The choice can be explained by things we lacked and had at the moment.

**We lacked:**
* Practical experience with Neural Networks using any software
* Expertise in Image Segmentation

**We had:**
* A strong desire to get, what we lacked
* Decent theoretical understanding of standard machine learning concepts
* Practical experience with standard statistical machine learning techniques, e.g. k-nearest-neighbours, linear regression and its modifications, support vector machines, etc.
* Theoretical understanding of how Neural Networks work, at the level of being comfortable with chapters 5-9 of "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016). http://www.deeplearningbook.org/

### Project Results:
* Achieved **top 10%** of the competition's leaderboard (for old competitions the results of late submissions are not displayed)
* Created a customisable U-net-like neural network with **160 different configurations**
* Gained practical experience of creating software for Image Segmentation
* Gained experience of doing independent research and working with academic papers

## Initial Research

Before starting any practical work, a few papers were chosen as preliminary reading.

### Initial papers:
* Qingqing Cui, Peng Pu, Lu Chen, Wenzheng Zhao, Yu Liu (2018). "Deep Convolutional Encoder-Decoder Architecture for Neuronal Structure Segmentation". https://ieeexplore.ieee.org/document/8698405
* Julián Gil González, Mauricio A. Álvarez, Álvaro A. Orozco (2015). "Automatic segmentation of nerve structures in ultrasound images using Graph Cuts and Gaussian processes". https://ieeexplore.ieee.org/document/7319045
* Julián Gil González, Mauricio A. Álvarez, Álvaro A. Orozco (2016). "A probabilistic framework based on SLIC-superpixel and Gaussian processes for segmenting nerves in ultrasound images". https://ieeexplore.ieee.org/document/7591636

Having little experience in the field, I found myself reading more papers, referenced in the original selection. I list them below for your interest.

### Follow-up papers:
* Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
* Evan Shelhamer, Jonathan Long, Trevor Darrell (2016). "Fully Convolutional Networks for Semantic Segmentation". https://arxiv.org/abs/1605.06211
* Fisher Yu, Vladlen Koltun (2016). "Multi-Scale Context Aggregation by Dilated Convolutions". https://arxiv.org/abs/1511.07122
* Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich (2014). "Going Deeper with Convolutions". https://arxiv.org/abs/1409.4842
* Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015). "Rethinking the Inception Architecture for Computer Vision". https://arxiv.org/abs/1512.00567v3
* Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi (2016). "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning". https://arxiv.org/abs/1602.07261v2

I also found the article which (in my opinion) very well summarises and explains the main concepts of the last three papers:
* Bharath Raj (2018), A Simple Guide to the Versions of the Inception Network. https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

## Acknowledgements

Due to the lack of practical experience, I based my project on the code written by Marko Jocić and Edward Tyantov.

**Marko Jocić's work:** [Kaggle][MJ's Kaggle], [GitHub][MJ's GitHub]

[MJ's Kaggle]: https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/21358#latest-651570
[MJ's GitHub]: https://github.com/jocicmarko/ultrasound-nerve-segmentation

**Edward Tyantov's work:** [Kaggle][ET's Kaggle], [GitHub][ET's GitHub]

[ET's Kaggle]: https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/22958#latest-132645
[ET's GitHub]: https://github.com/EdwardTyantov/ultrasound-nerve-segmentation

Marko Jocić released the code at the very beginning of the competition. I decided that using this code was something many people, including some industry professionals as Edward Tyantov, did during the competition. The code you see here is a modified combination of both sources.

### Reasons for using some blocks of code completely unchanged:
* My focus for this project was on machine learning, so I decided that I would use the data preparation process, written by professionals, after fully understanding it. The same applies to some other parts.
* Marko Jocić's code was available from the beginning of the competition, meaning that even the winner could potentially use the data preparation part and proceed to the other parts of the competition straight away.

### Reasons for modifications:
* I wanted to try different types of U-net-like architectures and not just replicate other people's work.
* I found Edward's code too complicated at times and wanted to have the code, which I could write myself from the very beginning. Due to this reason, I was simplifying and modifying the code where possible, without changing the final result.
* Due to some reason, Marko Jocić's code did not give me the 0.57 result, as stated in his Kaggle notebook.
* Finally, as you hopefully already read in my "Warning" section, I needed to be able to run the code on Kaggle servers, which was not possible, given the original code. The code by both authors did not compile. This happened because of the changes made to the Python libraries since 2016 when the competition was held. Even after arranging all the code together and fixing the compilation problem, the code had many bugs, occurring due to other Python library updates. Some of the bugs were left unchanged in author's GitHub versions.

## Running the code

### Running project's code

#### Running on Kaggle machines
If you are reading this in Kaggle notebook, you do not need to follow the next link. If not, to run the code on Kaggle machines, you can fork and run the notebook available through this [link.](https://www.kaggle.com/gbatchkala/urss-2019-project-review)

Some modules take several minutes to run. To execute a specific module, you will either have to set its execute parameter to **True**, or set **execute_all** parameter to **True**, which you can find in the next code block. The former executes a specific module, while the latter allows executing all modules.

You can also run this code in separate Kaggle script, which is just a concatenated version of all code in this notebook. The script code is available through these links: [Kaggle](https://www.kaggle.com/gbatchkala/urss-final-code-script), [GitHub](https://github.com/GeorgeBatch/ultrasound-nerve-segmentation/blob/master/urss_final_code_script.py)

#### Running on personal machines
If you would like to work with the code presented below on your sown machine, I recommend cloning my GitHub repository. This way, you do not need to set up a directory. [Link to project's repository.](https://github.com/GeorgeBatch/ultrasound-nerve-segmentation) 

## Required set-up

### Setting up your directory

If you decided to download code in separate files, first, you need to set up your directory structure as shown below. The structure mimics Kaggle's directory structure. On Kaggle, your script/notebook is in the working directory by default, while any data you upload goes inside the input directory.

```
- working
  |
  ---- Edward_Tyantov_edited.py
  |
  ---- urss_final_code_script.py
  |
  ---- data.py
  |
  ----...
- input
  |
  - ultrasound-nerve-segmentation
   |
   ---- train
   |    |
   |    ---- 1_1.tif
   |    |
   |    ---- …
   |
   ---- test
        |
        ---- 1.tif
        |
        ---- …
```

### Requirements:

See [kaggle/python docker image](https://github.com/kaggle/docker-python)

Minimal information:
* Python >= 3.5
* Keras >= 2.0 
* Tensorflow backend for Keras
* Working with files: os, sys
* For run-length-encoding: itertools
* Working with arrays: numpy
* Working with images: skimage

If you are using Theano backend, check that the shape of the data is in the form (samples, rows, cols, channels). Otherwise, the code breaks.

### Executing files:

To run this code, you need access to a GPU processing unit. I trained the model on Kaggle's GPU. Otherwise, it can take up to 2.5 days on Intel-i7 processors.

Set model configuration:
* check_pars.py
* configuration.py **- configure your network and the learning-rate optimizer**

Order of file execution:
* data.py
* train.py
* submission.py

Alternatively execute one of:
* urss_final_code_script.py
* Edward_Tyantov_edited.py

### Configuration
There are several versions of the U-net architecture you can try. If you want to try it out, do not change anything in the configuration module (configuration.py on GitHub) and you get the U-net kindly provided by Marko Jocić at the beginning of the competition.

In case you want to experiment, I list the versions I tried here. To configure your version of the U-net, you need to make decisions on each level of granularity (see below) combining all the top-level decisions into parameters, which you pass to the U-net-generating function. Below you can see the configuration structure:

* Number of outputs
    * One output
    * Two outputs
* Activation
    * ReLU
    * ELU
* Blocks for capturing information
    * Convolution blocks
        * Simple convolutions (see [simple implementation][simple u-net implementation])
            * With batch-normalization
            * Without batch-normalization
        * Dilated convolutions (see [referenced paper][Dilated convolutions paper])
            * With batch-normalization
            * Without batch-normalization
    * Inception blocks (see [the article][Inception-blocks article] and two referenced papers [paper 1][Inception v1], [paper 2][Inception v2 and v3])
         * Inception block v1, versions a, b
         * Inception block v2, versions a, b, c
         * Inception block et, versions a, b
* Skip connections from the down-path to the up-path of the U-net
    * Standard connections from the [original U-net paper]
    * Residual connections mimicking ResNet skip connections (see [paper][Inception v4, Inception-ResNet])
* Pooling layers reducing the size of the image
    * Non-trainable: Max-pooling layers
    * Trainable: Normalized Convolution layers with strides

**Optimizer**: Select any available optimizer from [Keras optimizers](https://keras.io/optimizers/)
    
[Original U-net paper]: https://arxiv.org/abs/1505.04597
[simple u-net implementation]: https://github.com/zhixuhao/unet

[Dilated convolutions paper]: https://arxiv.org/abs/1511.07122

[Inception-blocks article]: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
[Inception v1]: https://arxiv.org/abs/1409.4842
[Inception v2 and v3]: https://arxiv.org/abs/1512.00567v3
[Inception v4, Inception-ResNet]: https://arxiv.org/abs/1602.07261v2


### Running Edward Tyantov's code

As a beginner, I found Marko Jocić's code and the instructions for running it accessible. So I did not document the code and do not intend publishing it here in the future. At the same time, I had many problems with trying to make Edward Tyantov's code run correctly. This is why I made a concatenated, simplified, modified, and documented version of Edward Tyantov's original code available on Kaggle and GitHub:
* Kaggle: https://www.kaggle.com/gbatchkala/edward-tyantov-edited-py
* GitHub: https://github.com/GeorgeBatch/ultrasound-nerve-segmentation/blob/master/working/Edward_Tyantov_edited.py

In both versions, you can find the information about the changes, acknowledgements, and licence at the beginning of the python script.

For GitHub version see LICENCE. If you find any mistakes or want to update the code to satisfy current kaggle environment, please submit your changes to the file via pull request to my GitHub repository.


# Modules

## Check parameters module

File name: check_pars.py

Instructions:
* If used as as separate module, the "separate-module imports" part needs to be uncommented
* Do not change this module unless you want to make modifications to the u-net configuration


## Configuration module

File name: configuration.py

Instructions:
* If used as as separate module, the "separate-module imports" part needs to be uncommented
* Select a network configuration of your choice and record it in PARS (check BEST_PARS in check_pars module for format)
* Select any available optimizer from [Keras optimizers](https://keras.io/optimizers/)


## Data module

File name: data.py

Instructions:
* if used as as separate module, the "separate-module imports" part needs to be uncommented
* change execute_data to True 

Credits: Edward Tyantov

Modifications:
* Add get_nerve_presence(), load_nerve_presence() functions to allow one-output architectures
* Make appropriate updates, so the code can be run on Kaggle with current library versions
* Add documentation to all functions


## Metric module

File name: metric.py

Instructions: if used as as separate module, the "separate-module imports" part needs to be uncommented

Credits: Edward Tyantov

Modifications:
* Make appropriate updates, so the code can be run on Kaggle with current library versions
* Add documentation to all functions


## U-model blocks module

File name: u_model_blocks.py

Instructions: if used as as separate module, the "separate-module imports" part needs to be uncommented

Credits: Edward Tyantov

Functionality kept from Edward Tyantov's version or insignificantly modified:
* NConv2D()
* _shortcut()
* rblock
* inception_block_et()

New functionality:
* convolution_block()
* dilated_convolution_block()
* inception_block_v1()
* inception_block_v2()
* pooling_block()
* information_block()
* connection_block()


## U-model module

File name: u_model.py

Instructions: if used as as separate module, the "separate-module imports" part needs to be uncommented

Credits: Marko Jocić, Edward Tyantov

Modification: Make the architecture fully customisable


## Train module

File name: train.py

Credits: Marko Jocić

Modifications: allow for training with 1 or 2 outputs of the U-nel-like architecture

Instructions: If used as as separate module, the "separate-module imports" part needs to be uncommented


## Submission module

File name: submission.py

Credits: Edward Tyantov

Modifications: allow for training with 1 or 2 outputs of the U-nel-like architecture

Instructions: If used as as separate module, the "separate-module imports" part needs to be uncommented
