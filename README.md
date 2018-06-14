# Classification of encrypted traffic using deep learning
This repository contains the code used and developed during a master thesis at DTU Compute in 2018.  
Professor [Ole Winther](http://cogsys.imm.dtu.dk/staff/winther/) has been supervisor for this master thesis.  
Alex Omø Agerholm from [Napatech](https://www.napatech.com/) has been co-supervisor for this project.

In this thesis we examine and evaluate different ways of classifying encrypted network traffic using neural networks. For this purpose we create a dataset with a streaming/non-streaming focus. The dataset comprises seven different classes, five streaming and two non-streaming.  
The thesis serves as a preliminary proof-of-concept for Napatech A/S. 

The most desirable approach for Napatech is to classify individual packets upon arrival. Our experiments show that this approach is infeasible due to the content being encrypted and therefore without any distinguishable information. We therefore propose a novel approach where the unencrypted parts of network traffic, namely the headers are utilized. This is done by concatenating the initial headers from a session thus forming a signature datapoint as shown in the following figure: 

<img src="https://saliklp.github.io/plots/Header-datapoint.png" alt="Header datapoint" width="50%">

We explored the dataset by running t-SNE on the concatenated headers dataset. As can be seen in the figure below the classes  

<img src="https://saliklp.github.io/plots/t-SNE_16headers_all_merged_perplexity30.png" alt="t-SNE plot" width="50%">

In experiments using the header-based approach we achieve very promising results, showing that a simple neural network with a single hidden layer of less than 50 units, can predict the individual classes with an accuracy of 96.4\% and an AUC of 0.99 to 1.00 for the individual classes, as shown in the following figures.

<img src="https://saliklp.github.io/plots/trainAllMerged_acc964.png" alt="Confusion matrix of all 7 classes" width="40%"> <img src="https://saliklp.github.io/plots/trainAll_testAll_acc09990_merged_stream_nostream.png" alt="Confusion matrix of streaming vs non streaming" width="40%">
<img src="https://saliklp.github.io/plots/ROC_16header_12unit_all_.png" alt="ROC Plot" width="50%">

The thesis hereby provides a solution to network traffic classification using the unencrypted headers.
