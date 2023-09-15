# AnomalousDiffusion
Repository of all code for the Research Fellowship, summer 2023

The code contained in this repository is a compilation of all the code written by Alex Benitez for the King's Undergraduate Research Fellowship.

The objective of the fellowship was to first train a Convolutional Neural Network(CNN), to classify different types of trajectories for particles undergoing Anomalous Diffusion. In order to generate the trajectories, the andi-datasets package was used (https://github.com/AnDiChallenge/andi_datasets). A convolutional neural net was then trained on the data, finally, the innvestigate library (https://github.com/albermax/innvestigate)
was used to analyze the parts of trajectories which the neural nets visualized.

The next stage in development is to train a set of different CNN's, with different layer depth and kernel size combinations. This will be done in parallel in the create hpc cluster.
