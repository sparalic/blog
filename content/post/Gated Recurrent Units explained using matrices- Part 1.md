---
date: 2019-02-24T10:58:08-04:00
description: "Demystifying the math behind GRUs"
author: "Sparkle Russell-Puleri and Dorian Puleri"
featured_image: "/images/gruthumb.png"
tags: ["GRUs", "Deep Learning", "RNNs", "machine learning"]
title: "Gated Recurrent Units explained using matrices: Part 1"
---

by:[Sparkle Russell-Puleri](https://www.linkedin.com/in/sparkle-russell-puleri-ph-d-a6b52643/) and [Dorian Puleri](https://www.linkedin.com/in/dorian-puleri-ph-d-25114511/)<br>

Often times we get consumed with using Deep learning frameworks that perform all of the required operations needed to build our models. However, there is some value to first understanding some of the basic matrix operations used under the hood. In this tutorial we will walk you through the simple matrix operations needed to understand how a GRU works.

------

A detailed notebook can be found at: [Sparkle's](<https://github.com/sparalic/GRUs-internals-with-matrices>) or [Dorian's](<https://github.com/DPuleriNY/GRUs-with-matrices>) Github page.

Medium: [Medium post](https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18)

<strong> What is a Gated Recurrent Unit (GRU)?</strong><br>
Gated Recurrent Unit (pictured below), is a type of Recurrent Neural Network that addresses the issue of long term dependencies which can lead to vanishing gradients larger vanilla RNN networks experience. GRUs address this issue by storing “memory” from the previous time point to help inform the network for future predictions. At first glance, one may think that this diagram is quite complex, but it is quite the contrary. The intent of this tutorial is to debunk the difficulty of GRUs using Linear Algebra fundamentals.

![img](https://cdn-images-1.medium.com/max/1600/1*smlLUdHdARMBCW7EQAUttw.png)

The governing equations for GRUs are:

![img](https://cdn-images-1.medium.com/max/1600/1*lpe2VeZxdubIpwd8ZEVPIQ.png)Governing equations of a GRU

where z and r represent the update and reset gates respectively. While h_tilde and h represent the intermediate memory and output respectively.

<strong>GRUs vs Longterm Short Term Memory (LSTM) RNNs</strong><br>
The main differences between GRUs and the popular [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (nicely explained by [Chris Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)) are the number of gates and maintenance of cell states. Unlike GRUs, LSTMs have 3 gates (input, forget, output) and maintains an internal memory cell state, which makes it more flexible, but less efficient memory and time wise. However, since both of these networks are great at addressing the vanishing gradient problem required for efficiently tracking long term dependencies. Choosing between them is usually done using the following rule of thumb. When deciding between these two, it is recommended that you first train a LSTM, since it has more parameters and is a bit more flexible, followed by a GRU, and if there are no sizable differences between the performance of the two, then use the much simpler and efficient GRU.

------

<strong> Approach</strong><br>
To further illustrate the elegance of RNNs, we are going to walk you through the basics of linear algebra needed to understand the inner workings of a GRU. To do this we will use a small string of letters to illustrate exactly how the matrix calculations we take for granted, using pre-packaged wrapper functions, created many of the common DL frameworks. The point of this tutorial is not to set us back, but to help drive a deeper understanding of how RNNs work using Linear Algebra.

Sample using the following sample string as our input data:


text = 'MathMathMathMathMath'


However, algorithms are essentially mathematical equations of some sort, therefore our original text will have to be represented in numerical form before presenting it to the GRU layers. This is done in the following pre-processing step below.

------

<strong> Data Pre-processing</strong><br>
In the first step a dictionary of all of the unique characters is created to map each letter to a unique integer:

Character dictionary : {‘h’: 0, ‘a’: 1, ‘t’: 2, ‘M’: 3}

Our encoded input now becomes:
MathMath = [3, 1, 2, 0, 3, 1, 2, 0]

{{< gist sparalic 77f9b0df992d84515be06591427f20cb >}}

Let’s assume we want the following parameters:
1. Batch size (B) = 2 
2. Sequence length (S) = 3
3. Vocabulary (V) = 4
4. Output (O) = 4

![img](https://cdn-images-1.medium.com/max/1600/1*z5bPNW_2H37LomSEDDoYaA.png)

**So what is the time series?**
If you do a basic search for an RNN the image below is typically what you will find. This image is a generalized view of what happens in unfolded form. However, what does the x_(t-1), x_(t) and x_(t+1) (highlighted in red) mean in terms of our batches?

![img](https://cdn-images-1.medium.com/max/1600/1*vy7vsk4c0RJZqDlzqL_RjQ.png)Vanilla RNN architecture

In the case of our mini-batch, the time series represents each sequence with information flowing from left to right as shown in the figure below.

![img](https://cdn-images-1.medium.com/max/1600/1*sbmFYQG7vtO_qK9XWZmNyQ.png)Schematic of data flow for each one-hot encoded batch

<strong> Dimensions of our dataset</strong>

![img](https://cdn-images-1.medium.com/max/1600/1*6ORf-jUlU2tfR_G_LWiHmw.png)Batch anatomy

<strong>Step 1: Illustrated in code</strong>

{{< gist sparalic 6b1ae9d1f9406f0fb9d1ab9ee56558b3 >}}


After reshaping, if you check the shape of X you would find that you get a rank 3 tensor of shape: 3 x 3 x 2 x 4. What does this mean?

![img](https://cdn-images-1.medium.com/max/1600/1*m-DJs-oHAyARG1V5KqdTOQ.png)Dimensions of the dataset

<strong> What will be demonstrated?</strong><br>
The data is now ready for modeling. However, we want to highlight the flow of this tutorial. We will demonstrate the matrix operations performed for the first sequence (highlighted in red) within batch 1 (shown below). The idea is to understand how the information from the first sequence gets passed to the second sequence and so on.

![img](https://cdn-images-1.medium.com/max/1600/1*6Bz8RzixRbDEBAyefQUwaQ.png)Sequence used for walk through (Sequence 1 batch 1)

To do this we need to first recall how these batches are fed into the algorithm.

<strong> What will be demonstrated?</strong><br>
The data is now ready for modeling. However, we want to highlight the flow of this tutorial. We will demonstrate the matrix operations performed for the first sequence (highlighted in red) within batch 1 (shown below). The idea is to understand how the information from the first sequence gets passed to the second sequence and so on.

![img](https://cdn-images-1.medium.com/max/1600/1*6Bz8RzixRbDEBAyefQUwaQ.png)Sequence used for walk through (Sequence 1 batch 1)

To do this we need to first recall how these batches are fed into the algorithm.

![img](https://cdn-images-1.medium.com/max/1600/1*FfJofbmTdJrCuba0UwVF5w.png)Schematic of batch one being ingested into the RNN

More specifically, we will walk through all of the matrix operations done in a GRU cell for sequence 1 and the resulting outputs y_(t-1) and h_t will be calculated in the process (shown below):

![img](https://cdn-images-1.medium.com/max/1600/1*QEErOH5S8fs23SJ3D7TmdA.png)First time step of of batch 1

<strong> Step 2: Define our weights matrices and bias vectors</strong><br>
In this step we will walk you through the matrix operations used to calculate the z gate, since the calculations are exactly the same for the remaining three equations. To help drive this point home we are going to walk through the dot product of the reset gate z by breaking the inner equation down into three sections and finally we will apply the sigmoid activation function to the output to squish the values between 0 and 1:

![img](https://cdn-images-1.medium.com/max/1600/1*Jmkz2f-TbHk8sxuFY1c6ag.png)Reset gate

But first let’s define the network parameters:

{{< gist sparalic 0fe816ad24034923b754449fb46007c7 >}}

<strong> What is a hidden size?</strong><br>
The hidden size defined above, is the number of learned parameters or simply put, the networks memory. This parameter is usually defined by the user depending on the problem at hand as using more units can make it more likely to over fit the training data. In our case we chose a hidden size of 2 to make this easier to illustrate. These values are often initialized to random numbers from the normal distribution, which are trainable and updated as we perform back-propagation.

![img](https://cdn-images-1.medium.com/max/1600/1*7fTZ00KiaDR6G6Jeqv5a6A.png)Anatomy of the Weight matrix

<strong> Dimensions of our weights</strong><br>
We will walkthrough all of the matrix operations using the first batch, as it’s exactly the same process for all other batches. However, before we begin any of the above matrix operations, let’s discuss an important concept called broadcasting. If we look at the shapes of batch 1 (3 x 2 x 4) and the shape of Wz (4 x 2), the first thing that may come to mind is, how would we perform element-wise matrix multiplication on these two tensors with different shapes?

The answer is we use a process called “Broadcasting”. Broadcasting is used to make the shapes of the these two tensors compatible, such that we can perform our element-wise matrix operations. This means that Wz will get broadcasted to a non-matrix dimension, which in our case is our sequence length of 3. This then means that all of the other terms in the update equations z will also get broadcasted. Therefore, our final equation will look like this:

![img](https://cdn-images-1.medium.com/max/1600/1*woo19rEGH6U3wQ4IlO1pcA.png)Equation for z with weight matrices broadcasted

Before we perform the actual matrix arithmetic let’s visualize what sequence 1 from batch one looks like:

![img](https://cdn-images-1.medium.com/max/1600/1*JFNb6tluo-JGkyZufMGBuQ.png)Illustration of matrix operations and dimensions for the first sequence in batch 1

<strong> The update gate: z </strong><br>
The update gate determines how useful past information is to the current state. Here, the use of the sigmoid function results in update gate values between 0 and 1. Therefore, the closer this value is to 1 the more we incorporate past information, while values closer to 0 would mean that only new information is kept.

**Now let’s get to the math…**First term: Note that when these two matrices are multiplied using the dot product, we are multiplying each row by each column. Here, each row (highlighted in yellow) of the first matrix ( x_t) gets multiplied element-wise by each column (highlighted in blue) of the second matrix (Wz).

Term 1: Weights applied to the inputs

![img](https://cdn-images-1.medium.com/max/1600/1*9Ss6VEzwmP1ei8RYivKlzw.png)Dot product of the first term in the update gate equation

Term 2: Hidden Weights

![img](https://cdn-images-1.medium.com/max/1600/1*weIcsxqGJnaxsoNFaZewbg.png)Dot product of the second term in the update gate equation

Term 3: Bias Vector

![img](https://cdn-images-1.medium.com/max/1600/1*GYGO7dE1A3l9DkmudQkd5g.png)Bias vector

<strong> Putting it all together: z_inner</strong><br>

![img](https://cdn-images-1.medium.com/max/1600/1*qtMvjDndGbwEqna-T23jZg.png)Inner linear equation of the reset gate

The values in the resulting matrix is then squished between 0 and 1 using the sigmoid activation function:

![img](https://cdn-images-1.medium.com/max/1600/1*kAa09xCse5dowrXRBwpgpg.png)Sigmoid equation

<strong> The reset gate: r</strong><br>

Reset gate allows the model to ignore past information that might be irrelevant in future time-steps. Over each batch, the reset gate will re-evaluate the combined performance of prior and new inputs and reset as needed for the new inputs. Again because of the sigmoid activation function, values closer to 0 would mean that we would keep ignore the previous hidden state, and the opposite is true for values closer to 1.

![img](https://cdn-images-1.medium.com/max/1600/1*pbLxrzrelcELY_Bq1mzxkg.png)Reset gate

<strong> Intermediate Memory: h_tilde</strong><br>

The intermediate memory unit or candidate hidden state combines the information from the previous hidden state with the input. Since the matrix operations required for the first and third terms are the same as what we did in z, we will only present the results.

![img](https://cdn-images-1.medium.com/max/1600/1*cUPqBMYcui6eIN72zy7fxQ.png)Intermediate/candidate hidden state

Second term:

![img](https://cdn-images-1.medium.com/max/2400/1*YzTUU0OyG2kKRu7iRMicqQ.png)Second term matrix operations

<strong> Putting it all together: h_tilde</strong>

![img](https://cdn-images-1.medium.com/max/1600/1*869TP2blPO_-3XUKcV-LGQ.png)Inner linear equation calculation

The values in the resulting matrix is then squished between 0 and 1 using the tanh activation function:

![img](https://cdn-images-1.medium.com/max/1600/1*u_WElHRCjG6jMDdTBK3zNw.png)Tanh activation function

Finally:

![img](https://cdn-images-1.medium.com/max/2400/1*5utBb4Ejs5QT8ct1qz2DhQ.png)Candidate hidden state output

<strong> Output hidden layer at time step t: h_(t-1)</strong>

![img](https://cdn-images-1.medium.com/max/1600/1*TGkQnEuZPXB1ZcEg_HKDig.png)Hidden state for the first time step

![img](https://cdn-images-1.medium.com/max/1600/1*2AGyiQE22of4KXhid3wRvA.png)Resulting matrix for hidden state at time step 1

<strong> How does the second sequence in batch 1 (time step x_t) information from this hidden state?</strong>

Recall, that h_(t-n) is first initialized to zeros (used in this tutorial) or random noise to begin the training after which the network would learn and adapt. But after the first iteration, the new hidden state h_t will now be used as our new hidden state and the calculations above are repeated for sequence 2 at time step (x_t). The image below demonstrates how this is done.

![img](https://cdn-images-1.medium.com/max/1600/1*ehVF89Rz6KQZUIpc_yzbSQ.png)Illustration of the new hidden state calculated in the above matrix operations

This new hidden state h_(t-1) will not be used to calculate the output ( y_(t+1)) and hidden state h_(t)of the second time step in the batch and so on.

![img](https://cdn-images-1.medium.com/max/1600/1*XjZQ1Gk2HrZVWpPpl7PIUg.png)Passing of hidden states from sequence1 to sequence 2

Below we demonstrate how the new hidden state h_(t-1) is used to calculate subsequent hidden states. This is typically done using a loop. This loop iterates over all of the elements within each the given batch to calculate both h_(t-1).

<strong> Code Implementation: Batch 1 outputs: h(t−1), h(t) and h(t+1)</strong>


{{< gist sparalic 943f5101f99cde0a035256b824625d1e >}}


<strong> What will be the hidden state for the second batch?</strong>

If you are a visual person, it can be seen as a series the output at h_(t+1), will then be feed to the next batch and the whole process begins again.

![img](https://cdn-images-1.medium.com/max/1600/1*JG7_O3CQ2xFU6cgbZ5ti5Q.png)Passing of hidden states across batches

<strong> Step 3: Calculate the out predictions for each time step</strong>

To obtain our predictions for each time step we first have to transform our output using a linear layer. Recall the dimensions of columns in the hidden states h_(t+n) is essentially the dimension of the network size/hidden size. However, we have 4 unique inputs and we are expecting our outputs to also have a size of 4. Therefore, we use what is called a dense layer or fully connect layer to transform our outputs back to the desired dimensions. This fully connected layer is then passed into an activation function (Softmax for this tutorial), depending on the desired output.

![img](https://cdn-images-1.medium.com/max/1600/1*CH98GHTUSY0tZ-HSeItwQQ.png)Fully connected/Linear layer

Finally, we apply the Softmax activation function to normalize our outputs into a probability distribution, which sums up to 1. The Softmax function:

![img](https://cdn-images-1.medium.com/max/1600/1*ZkDV0gqEimWUyxox1vSD3Q.png)Softmax equation

Depending on the textbook you may see different flavors of the softmax, particularly using the softmax max trick which subtracts the maximum value of the entire dataset to prevent exploding values for large y_lineary/fully_connected. In our case this means that our max value of 0.9021 will first be subtracted from y_linear prior to applying it the the softmax equation.

Let’s break this down, please note that we cannot subset the sequences as we did earlier because the summation requires all elements in the entire batch.

1. Subtract the max value of the entire dataset from all of the elements in the fully connected layer:

![img](https://cdn-images-1.medium.com/max/2400/1*vzrW_yc2M5NsW2LaCrOcnw.png)Applying the Max trick for Softmax equation

2. Find the sum of all of the elements within the matrix of exponents

![img](https://cdn-images-1.medium.com/max/1600/1*iwk8bL37avBamuvDsfBJpQ.png)Sum of the exponents for each row

![img](https://cdn-images-1.medium.com/max/1600/1*bABbwn6mD9g7hS4P0IflpQ.png)Final Softmax output for the first sequence in batch 1

<strong> Finally, training our network (forward only)</strong><br>
Here we train the network on the input batches by running each batch through the network several times, which is called an epoch. This allows the network to learn the sequences many times. This is then followed with a loss calculation and back-propagation to minimize our loss. In this section we will implement all of the code snippets showed above in one pass. Given the the small input size we will only demonstrate the forward pass, as the calculation of the loss function and back-propagation will be detailed in a subsequent tutorial.

{{<gist sparalic 5c872fa0bf406f75a1e3ccd18e00ef17>}}

This function will feed a primer of letters to the network help create an initial states and avoid making random guesses. As shown below the first couple of strings generated are a bit erratic, but after a few passes it seems to get at least the next two characters correct. However, given the small vocabulary size this network is most likely overfitting.

{{<gist sparalic c91c89f5bfcf2b77d075e00562975870>}}

<strong> Final Words</strong>

The intent of this tutorial was to provide a walkthrough of the inner working of GRUs using demonstrating how simple matrix operations when combined can make such a powerful algorithm.


<strong> References:</strong>

1. The Unreasonable Effectiveness of Recurrent Neural Networks
2. Udacity Deep Learning with Pytorch
3. fastai Deep Learning for Coders
4. Deep Learning — The Straight Dope
5. Deep Learning Book
6. [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
