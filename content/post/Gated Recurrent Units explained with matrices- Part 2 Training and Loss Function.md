---
date: 2019-03-05T10:57:08-04:00
description: "Demystifying the math behind GRUs"
author: "Sparkle Russell-Puleri and Dorian Puleri"
featured_image: "/images/ce.png"
tags: ["GRUs", "deep Learning", "RNNs", "deep learning", "machine learning", "cross entropy"]
title: "Gated Recurrent Units explained with matrices: Part 2 Training and Loss Function"
---
by: [Sparkle Russell-Puleri](https://www.linkedin.com/in/sparkle-russell-puleri-ph-d-a6b52643/) and [Dorian Puleri](https://www.linkedin.com/in/dorian-puleri-ph-d-25114511/)

Medium:[Medium Post](https://medium.com/@sparklerussell/gated-recurrent-units-explained-with-matrices-part-2-training-and-loss-function-7e7147b7f2ae) 

In [part one](https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18) of this tutorial series, we demonstrated the matrix operations used to estimate the hidden states and outputs for the forward pass of a GRU. Based on our poor results, we obviously need to optimize our algorithm and test it on a test set to ensure generalizability. This is typically done using several steps/techniques. In this tutorial we will walkthrough what happens under the hood during optimization, specifically calculating the loss function and performing backpropagation through time to update the weights over several epochs.

{{<gist sparalic c064aa09f7b0bac71764054313ed0df2>}}

<strong> What’s happening here?</strong><br>
As we pointed out in the first tutorial the first couple of strings generated are a bit erratic, but after a few passes it seems to get at least the next two characters correct. However, in order to measure how inconsistent our predictions are versus the true labels, we need a metric. This metric is call the loss function, that measures how well the model is performing. It is a positive value that decreases as the network becomes more confident with it’s predictions. This loss function for multi-class classification problems is defined as:



![img](https://cdn-images-1.medium.com/max/1600/1*hjQPgT1WdcVlXQs0yBxNjQ.png)

<center>Cross entropy equation</center>

Recall, our calculated hidden states and predicted outputs for the first batch? This picture seems a bit busy, however the goal here is to visualize what you outputs and hidden states actually look like under the hood. The predictions are probabilities which were calculated using the Softmax activation function.



![img](https://cdn-images-1.medium.com/max/1600/1*8r2d_Tqd0bbg5kYTe_DqSw.png)

GRU outputs with matrices

Let’s re-run the training loop storing the outputs ( y_hat) and hidden states (h_(t-1), h_t, and, h_(t+1)) for each sequence in batch 1.

<strong> Illustration in code:</strong><br>
To understand what is happening you will notice that we work from the inside out, before moving to functions. Here, we are grabbing the outputs and hidden states calculated with just two loops.
<script src="https://gist.github.com/sparalic/29d16dd1103580eea3f077dc2515ae11.js"></script>


The cross entropy loss is first calculated for each sequence in the batch then averaged over all sequences. So, in this example we will calculate the cross entropy loss for each sequence from scratch. But first, let’s grab the predictions made on the first batch. To do this we will grab the for element ( index 0) from our ht_2 and outputs variables.

<script src="https://gist.github.com/sparalic/dac4a7cbd061918feccde16729f84a41.js"></script>


<center>Model predictions and hidden states for the first batch</center>

<strong>How well did we perform?</strong>

By looking at the output probabilities we can tell that we did not do so well. However, let’s quantify it using the cross entropy equation! Here we will work our way from the inner term out on the first sequence in the batch. Note, the code will included all 3 sequences in batch 1.



![img](https://cdn-images-1.medium.com/max/1600/1*lUUmNbjMNS1rfX4El9i5VA.png)

**First term:** Element-wise multiplication of the true labels with the log of the predicted labels



![img](https://cdn-images-1.medium.com/max/1600/1*GA9celuV8C1zouFte1eE7Q.png)

<center>Cross entropy term 1 calculation</center>

<strong>Implementation in code:</strong>

<script src="https://gist.github.com/sparalic/d7170e6af90d48976a31f22cf1b3d14c.js"></script>

**Second term:**Summation of remaining values within each sequence. In this step, it is key to note that the axis will be reduced row-wise, only containing the non-zero terms. This will be done in a loop programatically.



![img](https://cdn-images-1.medium.com/max/1600/1*bOLRznERdlJ2tzFcmvy4UQ.png)

<center>Cross entropy term 2 calculation</center>

<strong>Implementation in code:</strong>

<script src="https://gist.github.com/sparalic/d5ba1b679576777c1c8b6a5b5f4bda0d.js"></script>

**Third term:** Mean of the reduced samples for first sequence within the batch tow-wise. This example calculation was done on the first sequence within batch 1. However, the code implementation covers all 3 sequences in batch 1.


![img](https://cdn-images-1.medium.com/max/1600/1*Xju6WqPDo5SalVFG5Nbk4A.png)
<center>Cross entropy term 3 calculation</center><br>

<strong>Implementation in code:</strong>

<script src="https://gist.github.com/sparalic/471b9d94c61ef7233b3fe5e055043162.js"></script>

<strong>Averaging the cross entropy losses of each sequence within batch 1</strong><br>
Note, in practice this step will be done over each mini-batch by keeping a running average of the losses for each batch. It essentially sums up what we calculated for the cross entropy (loss for each sequence in batch 1) and divides it by the number of sequences within the batch.

<script src="https://gist.github.com/sparalic/011208c8d34f4d618e97bd67bec2427f.js"></script>

<strong>How did we do?</strong><br>
<script src="https://gist.github.com/sparalic/7e3e9a8da6b2ae96c655581aee9b3879.js"></script>

<strong>Explanation</strong><br>
So we optimized reduced our loss and we are not predicting well…why? Well, as mentioned in the first tutorial this is an extremely small dataset, when training on a neural net made from scratch. It is recommended that you do so with lots of data. However, the purpose of this tutorial is not to create the high performance neural net, but to demonstrate what goes on under the hood.

<strong>Backpropagation</strong><br>
The final step involves a backward pass through the algorithm. This step is called backpropagation, and it involves understanding the impact of adjusting the weights on the cost function. This is done by calculating the error vectors delta starting from the final layer backward by repeatedly applying the chain rule through each layer. For more detailed proof of back-prop through time: <https://github.com/tianyic/LSTM-GRU/blob/master/MTwrtieup.pdf>

<strong>References:</strong><br>
1. The Unreasonable Effectiveness of Recurrent Neural Networks
2. Udacity Deep Learning with Pytorch
3. Fastai Deep Learning for Coders
4. Deep Learning — The Straight Dope (RNNs)
5. Deep Learning Book