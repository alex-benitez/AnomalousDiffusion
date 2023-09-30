# AnomalousDiffusion
(Despite what github might claim, this code is written in Python, not PureBasic)

Repository of all code for the Research Fellowship, summer 2023

The code contained in this repository is a compilation of all the code written by Alex Benitez for the King's Undergraduate Research Fellowship.

The objective of the fellowship was to first train a Convolutional Neural Network(CNN), to classify different types of trajectories for particles undergoing Anomalous Diffusion. In order to generate the trajectories, the andi-datasets package was used (https://github.com/AnDiChallenge/andi_datasets). A convolutional neural net was then trained on the data, finally, the innvestigate library (https://github.com/albermax/innvestigate)
was used to analyze the parts of trajectories which the neural nets visualized.

The next stage in development is to train a set of different CNN's, with different layer depth and kernel size combinations. This will be done in parallel in the create hpc cluster.

There are some difficulties in getting the libraries to work together, since tensorflow and andi_datasets require different versions of numpy. Therefore if you want to run the networks for yourself, once the training is done, I will upload and outline the most relevant ones to /Models/All_The_Nets. The basic one I used for experimentation and all of the plots is /Models/Convolutional_Final.

## The programs:
**Automation.py:** This file, generates the networks found within /Models/All_The_Nets, first it generates the set of numbers that go 123 132 123123 123132..., and then generates a network for kernel sizes 2,4,6,10,16,32. What the numbers indicate is the order of the layers and the number of layers. If the program reads a 1, it adds a convolutional layer, if it finds a 2, it adds a pooling layer, and if it finds a 3 it adds a batch normalization. I alternated them in order to see if there is any actual difference in the order they are in.

**ConfusionMatrix.py:** It simlpy generates a confusion matrix for a given network, by generating 2000 of each trajectory and recording the accuracy (this is quite resource intensive, so you can use less trajectories if needed).

![heatmaplotsofvalues](https://github.com/alex-benitez/AnomalousDiffusion/assets/63901940/94e00ee7-b4de-40d8-9fc4-ae9df8b767b5)


**ConvolutionalDiffusion.py:** Is the program I originally used to tinker with CNN's, it contains the layer structure for the net I used for all of the graphs.

**Convtools.py:** Contains a set of useful tools to either generate trajectories or test the model.

**Heatmapsplot.py:** Using a library called LRP Toolbox, it generates a visualization of what parts of a trajectory for each type of trajectory, a given network focuses on.

![Fivegraphs2](https://github.com/alex-benitez/AnomalousDiffusion/assets/63901940/a02ba9e3-ec5b-404f-9240-94f764b62a95)

**TheBigOne.py:** Was an attempt to write a complete program that parallelizes the generation of datasets and training of All_The_Nets. It was unsuccseful because due to constraints in the create hpc cluster, I could not have both andi_datasets and tensorflow in the same environment, therefore I had to split it in two.

**Trajectories.py:** Generates a set of 60000 trajectories, puts 50k in one file and 10k in another for validation, it stores them in a folder named after the same network.

**Trainnets.py:** Child program for TheBigOne, it takes the dataset that is stored in the folder in Datasets which is named after the net structure. 
