

---
date: 2019-04-20T10:58:08-04:00
description: "Full pytorch implementation Doctor AI paper using Electronic Health Records"
author: "Sparkle Russell-Puleri"
featured_image: "/images/healthai.jpeg"
tags: ["Healthcare", "Electronic Health Records","GRUs", "deep Learning", "RNNs", "deep learning", "machine learning"]
title: "Predicting future medical diagnoses with RNNs using Fast AI API from scratch"
---
by: Sparkle Russell-Puleri 

In the first [part one](https://towardsdatascience.com/using-electronic-health-records-ehr-for-predicting-future-diagnosis-codes-using-gated-recurrent-bcd0de7d7436) of this tutorial we created a rough template of the [Doctor AI: Predicting Clinical Events via Recurrent Neural Networks paper(2016)](https://arxiv.org/abs/1511.05942) by Edward Choi et.al. In this tutorial we took it a step further using the Fast.ai buttom up approach. This code is fully functional and details on how the data was processed can be accessed in [part one](https://towardsdatascience.com/using-electronic-health-records-ehr-for-predicting-future-diagnosis-codes-using-gated-recurrent-bcd0de7d7436).

------

<strong>Load Data</strong><br>

<strong>*About the data set:*</strong><br>
This study will utilize the [MIMIC III](https://mimic.physionet.org/) electronic health record (EHR) dataset, which is comprised of over 58,000 hospital admissions for 38,645 adults and 7,875 neonates. This dataset is a collection of de-identified intensive care unit stays at the Beth Israel Deaconess Medical Center from June 2001- October 2012. A detailed walkthrough of the data pre-processing steps used can be found in [part one](https://towardsdatascience.com/using-electronic-health-records-ehr-for-predicting-future-diagnosis-codes-using-gated-recurrent-bcd0de7d7436).

The data pre-processed datasets will be loaded and split into a train, test and validation set at a `75%:15%:10%` ratio.

<script src="https://gist.github.com/sparalic/75aacd24ad8fbebc8ba64ff678278d39.js"></script>
<center>Data loading function</center><br>

<strong>Padding sequences: to address variable length sequences</strong>
Using the artificial EHR data created in part one we pad the sequences to the length of the longest sequence in each mini-batch. To help explain this in greater depth let’s take a look at the `Artificial EHR data` created in part one.<br><br>


<strong>Detailed explanation using artificially generated EHR data</strong><br><br>

<script src="https://gist.github.com/sparalic/40881c158519c09ec4d8c74b0289077a.js"></script>

Here you can see that we have an array with two list, each list representing a unique patient. Now, within each list are a series of lists, each representing a unique visit. Finally, the encoded numericals represent the diagnosis codes assigned during each unique visit. It is key to note that given the uniqueness of each patient’s condition, there are `variable length` sequences for both the visits and diagnosis codes assigned. Because EHR data is longitudinal in nature and we are often interested in understand a patient's risk or progression over time. When using tabular data processing these nested time-dependent `variable length` sequences can get complicated quickly. Recall the following image from part one, detailing the mapping of each visit date to the diagnosis codes assigned during that visit.

![img](https://cdn-images-1.medium.com/max/1600/1*poQbXNnKQlEPZq7q-ZWQFg.png)
<center>Patient Sequence Encodings</center>

<script src="https://gist.github.com/sparalic/71f4fe2465e890e43c74f1090ee858ad.js"></script>

![img](https://cdn-images-1.medium.com/max/1600/1*Fgegl5Xn5ZNBwsSv16hREw.png)
<center>Python Pickled List of List containing patient visits and encoded diagnosis codes</center>

<script src="https://gist.github.com/sparalic/957795c43bb0005f02a5fa70eca0f5f3.js"></script>
<center>Padding function</center><br>

<strong>So what exactly are we padding with this nested list?</strong><br>

Let’s break down the padding function:<br>

1. `lenghts = np.array([len(seq) for seq in seqs]) - 1` Here were are mysteriously subtracting 1 from the length, in the author's notes he mentioned that both the `visit` and `label` files must match as the algorithm takes care of the time lag for inference time.

What does this mean? Given the structure of the data, the last visit in each patient’s record will be removed. As illustrated here:<br><br>

{{< figure src="/images/Lengths.png" title="" >}}
<center>Removing the last visit for inference</center><br>

<strong>Aside: Dealing with variable length sequences in a Character level RNN</strong><br>
If this was a character level problem let’s say [`Sparkle`,`Dorian`, `Deep`, `Learning`]. The sequences are first arranged by length, in descending order and padded with zeros (red), where each letter represents a token. As shown here:<br>

{{< figure src="/images/test_seq.png" title="" >}}
<center>Variable length sequence padding</center><br>

<strong>EHR data:</strong><br>
However, for EHR data of this form given our current problem, instead of each encoded diagnosis code representing a unique token. In this case, each visit represents a token/sequence. So, using the same approach used with character level RNNs we first arrange each mini-batch by the patient visits in descending order. In this the patient 1 has the longest visit history with a total of two visits, while patient 2’s visits will be padded to the max length of 2, since it’s the longest sequence. As shown here:<br>

{{< figure src="/images/padding1.png" title="" >}}
<center>Padding EHR data</center><br>

Now, that we have taken care of the variable length problem, we can proceed to multi-one hot encode our sequences. This will result in the desired dimensions of S x B x I ( Sequence length, Batch size, Input dimensions/vocab).<br><br>

Here we can easily see that the sequences will represent the patient with the longest visit history in each mini-batch, while all others will be padded to this length (red). Depending on the desired batch size, the batch size will represent how many patients sequences are feed in at each timestep. Finally, the inner list will be encoded to the length of the vocabulary, which in this case the number of unique diagnosis codes in the entire dataset.

{{< figure src="/images/minibatch.png" title="" >}}
<center>Multi-one hot encoded sequences</center>

<strong>Labels</strong><br>
To ensure that the labels are shifted over by one sequence, so that the algorithm can accurately predict the next time step. The author took care of this by ensuring that the training data excluded the last visit within each patient’s history, using this logic `for xvec, subseq in zip(x[:, idx, :], seq[:-1]):`, where we took all but the last visit within each patient's visit record `seq[:-1]`. For the labels, this meant that the sequences will start from the patients second visit, or in python's indexing style the first index `for yvec, subseq in zip(y[:, idx, :], label[1:])`, where the label `label[1:]`, is shifted by one.<br><br>

{{< figure src="/images/labels.png" title="" >}}<br>
<strong>Label time step lag</strong><br>

<strong>What is masking and what does it do?</strong><br>
Masking allows the algorithm to know where the true sequences are in one-hot encoded data, simply put ignore/filter out the padding values, which in our case are zeros. This allows us to easily handle variable length sequences in RNNs, which require fixed length inputs. How is it done? Remember the `lengths` variable? This variable stores the effective lengths of each patient's sequences in descending order (recall: after removing the last sequence in each record for inference, eg. patient 1 has 3 visits, but length will reflect only 2). The logic in the code `mask[:lengths[idx], idx] = 1.` then fills in our zeroed tensor along the rows with 1's to match the length of each patient sequence from largest to smallest.<br><br>

lenghts_artificial → array([2, 1])<br><br>

mask_artificial → tensor([[1., 1.], [1., 0.]])<br><br>

<strong>Data Loaders and Sampler</strong><br>
The `Dataset` class is an abstract class that represents the data in x and y pairs.

<script src="https://gist.github.com/sparalic/71a44d7ceb30f7aae1921c856a811e3f.js"></script>

The `Sampler` class randomly shuffles the order of the training set (validation set will not be randomized). Additionally, it keeps the exact amount of sequences needed created a full batch.

<script src="https://gist.github.com/sparalic/a976e8ab1af40b5961736a22948457c8.js"></script>

The `DataLoader` class combines the dataset and the data sampler which iterates over the dataset and grabs batches.

<script src="https://gist.github.com/sparalic/3eab3f4c4b0c25ed58853d4a24b42679.js"></script>

<strong>Embedding Layer</strong><br>
The `Custom_Embedding` class was used to project the high-dimensional multi-hot encoded vectors to a lower dimensional space prior to presenting the input data to the GRU. In this step the auther used two approaches

1. Random initialization , then learn the appropriate W(emb)W(emb) weights during back-prop

![img](https://cdn-images-1.medium.com/max/1600/1*28q4wxkSNNC2NqB6-dPptA.png)

2. Pre-trained embedding initialized using the Skip-gram algorithm, then refine weights during back-prop

![img](https://cdn-images-1.medium.com/max/1600/1*MVrc4Z8R9667Ll7Oyk55sQ.png)

In this implementation of the paper we used the first approach. Therefore, the `Custom Embedding` class was created to created apply a tanh activation on the embedding layer.

<script src="https://gist.github.com/sparalic/8f62498cab8df9443f2d9521d5c0b4c3.js"></script>

<strong>Dropout Layer</strong><br>
In this paper the author used the naive application of dropout that was first introduced by [Srivastava (2014)](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf). While this method works well, it impacts the RNNs ability to retain long term dependencies, because we are not maintaining the same mask across each timestep. Why is this important? It’s simple, if we randomly sample a new mask at each time step, it perturbs our RNNs connections making it difficult for the network to determine what information might be relevant in the long term. In this approach, I tested the a technique proposed by Gal and Ghahramani (2016) and further developed by [Merity (2017)](https://arxiv.org/pdf/1708.02182.pdf) for LSTMs. Here, they proposed overcoming the aforementioned problem with associated random sampling, by using the same dropout mask across multiple time steps in LSTMs. Here, I will applied the same approach on a GRU between each layer (two layers).

<script src="https://gist.github.com/sparalic/e1c56da2f8f5a0f999732512456a29a3.js"></script>

<strong>Doctor AI: Predicting Clinical Events via Recurrent Neural Networks</strong><br>
Despite the popularity and preference given to LSTMs. This paper used a GRU architecture, for its simplicity and ability to get similar performance as LSTMs. The dataset used in this paper contained `263, 706 patients`, whereas our dataset (MIMIC III) contained a total of `7537 patients`. However, the author demonstrated transfer learning can be a viable option in cases where one hospital system lack the large scale datasets need to train deep learning models like Dr. AI. Using the following architecture, my interest lies in the prediction of the patient's future diagnosis codes. However, one can easily extrapolate the algorithm to predict both diagnoses and duration between visits.

![img](https://cdn-images-1.medium.com/max/1600/1*NAT-F4V9OkG8e6uPpaoM1A.png)
<center>Model Architecture</center>

<script src="https://gist.github.com/sparalic/d6126a418bb94b28adec640a41b35528.js"></script>

<strong>GRU Layer:</strong><br>
This class uses the `EHR_GRU` cell class and allows the iteration over the desired number of layers.

<script src="https://gist.github.com/sparalic/f2e0e0cdd5c2d3d40fd60a66597d8e55.js"></script>

<strong>Loss Function:</strong><br>
The loss function used to assess model perform, contained a combination of the cross entropy. The prediction loss for each mini-batch was normalized to the sequence length. Finally, L2-norm regularization was applied to all of the weight matrices.

<script src="https://gist.github.com/sparalic/8a09f94c8fbb93519436d30352b9c1b7.js"></script>

<strong>Model Parameters:</strong><br>
The parameters used here were selected from those used in the Dr AI paper. The major difference between this approach and what I present here, was my use of the more updated drop out approach for RNNs.

numClass = 4894 
inputDimSize = 4894
embSize = 200
hiddenDimSize = 200
batchSize = 100 numLayers = 2

<strong>Load Data:</strong><br>
It’s key to note that you want to pass in the same file for the sequences and labels into the `load_data` function, as the model will take care of the adjusting the time steps for prediction internally.

<script src="https://gist.github.com/sparalic/6b11576cf8ff4dae15ef98b9665a332d.js"></script>

<strong>Training and validation loop</strong>

<script src="https://gist.github.com/sparalic/4a7486ee28de9baa711c5236026a123c.js"></script>

<strong>Comparison of my implementation to the paper’s algorithm:</strong><br>
I ran the same sequences on the paper’s algorithm, which is written in theano and python 2.7 and here you can see that the best cross entropy score after 10 epochs is about 86.79 vs. my 107. While, I am not performing better with some more hyperparameter tuning and optimization the algorithm can definitely perform better.

![img](https://cdn-images-1.medium.com/max/1600/1*nVTENwWAKdkN844Z5HFgzQ.png)
<center>Dr. Algorithm results for comparison</center>

<strong>Observations:</strong><br>
As you can see our training and validation losses are about the same, with such a small subset of the data used in the actual paper. It might be difficult to get better performance without overfitting. However, the intent of this tutorial was to provide a detailed walkthrough of how one can use EHR data to drive insights!

<strong>Full Script</strong>
[Github]()

<strong>Next Steps:</strong>
Add Callbacks using Fast.AI’s callback approach to track in training stats
Play around with different initialization approaches

<strong>Acknowledgements:</strong>
Fast.ai (Rachel Thomas, Jeremey Howard, and the amazing fast.ai community)
Dorian Puleri
