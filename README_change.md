# Kaggle Ultrasound Nerve Segmentation Competition (2016)<br>as an Undergraduate Research Project.

# Current version is still work in progress

## Warning!!!

The code was run on Kaggle machines in July 2019 - October 2019 using Python 3 interpreter. It might stop working in the future due to the changes to Kaggle virtual environment and the updates of Python libraries used in this code. Please check these pages for more information:
* Kaggle documentation: https://www.kaggle.com/docs/kernels#the-kernels-environment
* kaggle/python docker image: https://github.com/kaggle/docker-python

## About the Project

### Project Details:

* Title: Machine learning in the service of surgeons
* Author: George Batchkala, g.batchkala@warwick.ac.uk
* Supervisor: Dr Sigurd Assing, s.assing@warwick.ac.uk
* Institution: University of Warwick
* Department: Statistics
* Project funding: Undergraduate Research Support Scheme at the University of Warwick
* Project dates: July 1st 2019 - August 29th 2019
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


### Initial Research

Before starting any practical work, a few papers were chosen as preliminary reading.

**Initial papers:**
* Qingqing Cui, Peng Pu, Lu Chen, Wenzheng Zhao, Yu Liu (2018). "Deep Convolutional Encoder-Decoder Architecture for Neuronal Structure Segmentation". https://ieeexplore.ieee.org/document/8698405
* Julián Gil González, Mauricio A. Álvarez, Álvaro A. Orozco (2015). "Automatic segmentation of nerve structures in ultrasound images using Graph Cuts and Gaussian processes". https://ieeexplore.ieee.org/document/7319045
* Julián Gil González, Mauricio A. Álvarez, Álvaro A. Orozco (2016). "A probabilistic framework based on SLIC-superpixel and Gaussian processes for segmenting nerves in ultrasound images". https://ieeexplore.ieee.org/document/7591636

Having little experience in the field, I found myselt reading more papers, referenced in the original selection. I list them below for your interest.

**Follow-up papers:**
* Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
* Evan Shelhamer, Jonathan Long, Trevor Darrell (2016). "Fully Convolutional Networks for Semantic Segmentation". https://arxiv.org/abs/1605.06211
* Fisher Yu, Vladlen Koltun (2016). "Multi-Scale Context Aggregation by Dilated Convolutions". https://arxiv.org/abs/1511.07122
* Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich (2014). "Going Deeper with Convolutions". https://arxiv.org/abs/1409.4842
* Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015). "Rethinking the Inception Architecture for Computer Vision". https://arxiv.org/abs/1512.00567v3
* Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi (2016). "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning". https://arxiv.org/abs/1602.07261v2

I also found the article which (in my opinion) very well summarises and explains the main concepts of the last three papers:
* Bharath Raj (2018), A Simple Guide to the Versions of the Inception Network. https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202


### Acknowledgements

Due to the lack of practical experience, I based my project on the code written by Marko Jocić (MJ) and Edward Tyantov (ET).


**Marko Jocić's work:**
* Kaggle: https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/21358#latest-651570
* GitHub: https://github.com/jocicmarko/ultrasound-nerve-segmentation      

**Edward Tyantov's work:**
* Kaggle: https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/22958#latest-132645
* GitHub: https://github.com/EdwardTyantov/ultrasound-nerve-segmentation

Marko Jocić released the code at the very beginning of the competition. I decided that using this code was something many people, including some industry professionals as Edward Tyantov, did during the competition. The code you see here is a modified combination of both sources.

**Reasons for using some blocks of code completely unchanged:**
* My focus for this project was on machine learning, so I decided that I would use the data preparation process, written by professionals, after fully understanding it. The same applies to some other parts.
* Marko Jocić's code was available from the beginning of the competition, meaning that even the winner could potentially use the data preparation part and proceed to the other parts of the competition straight away.

**Reasons for modifications:**
* I wanted to try different types of U-net-like architecture and not just replicate other people's work.
* I found Edward's code too complicated at times and wanted to have the code, which I could write myself from the very beginning. Due to this reason, I was simplifying the code where possible, without changing the final result.
* Due to some reason, Marko Jocić's code did not give me the 0.57 result, as stated in his Kaggle notebook.
* Finally, as you hopefully already read in my "Warning" section, I needed to be able to run the code on Kaggle servers, which was not possible, given the original code. The code by both authors did not compile. This happened because of the changes made to the Python libraries since 2016 when the competition was held. Even after arranging all the code together and fixing the compilation problem, the code had many bugs, occurring due to other Python library updates. Some of the bugs were left unchanged in author's GitHub versions.
