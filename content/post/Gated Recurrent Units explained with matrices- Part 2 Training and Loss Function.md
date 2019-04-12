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

```
#Input text

# This will be our input ---> x
text = 'MathMathMathMathMath'

# Training loop
max_epochs = 5  # passes through the data
for e in range(max_epochs):
    h = torch.zeros(batch_size, hiddenSize)
    for i in range(num_batches):
        x_in = X[i]
        y_in = y[i]

        out, h = gru(x_in, h)
        print(sample('Ma',20))

'''
MataMttataatthtthaMhMa
MattatttttattatatataMt
MattMtMttttthtMtatMtth
MaMtttttttttatttttatht
MattattMattttMtttthtMt
MattataahMtataatttttMt
MahMtMMtttMttMMaMMthaM
MatttMttttttMttttattah
MahMtMMhhatMaMattttaMt
MaMMtMMhtttththhMtthth
MaMthttMthhttaathtMMta
MaMMaaaahtMhhttatthtMt
MaataMtthMtMththtMMatt
MahhhtthMMthtMtatMtttt
MahtMhttttaattMttttMaa
'''
```


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

```
ht_2 = [] # stores the calculated h for each input x
outputs = []
h = torch.zeros(batch_size, hiddenSize) # intitalizes the hidden state
for i in range(num_batches):  # this loops over the batches 
    x = X[i]
    for i,sequence in enumerate(x): # iterates over the sequences in each batch
        z = torch.sigmoid(torch.matmul(sequence, Wz) + torch.matmul(h, Uz) + bz)
        r = torch.sigmoid(torch.matmul(sequence, Wr) + torch.matmul(h, Ur) + br)
        h_tilde = torch.tanh(torch.matmul(sequence, Wh) + torch.matmul(r * h, Uh) + bh)
        h = z * h + (1 - z) * h_tilde
        
        # Linear layer
        y_linear = torch.matmul(h, Wy) + by
        
        # Softmax activation function
        y_t = F.softmax(y_linear, dim=1)
        
        ht_2.append(h)
        outputs.append(y_t)
        
ht_2 = torch.stack(ht_2)
outputs = torch.stack(outputs)
```


The cross entropy loss is first calculated for each sequence in the batch then averaged over all sequences. So, in this example we will calculate the cross entropy loss for each sequence from scratch. But first, let’s grab the predictions made on the first batch. To do this we will grab the for element ( index 0) from our ht_2 and outputs variables.

```
hidden_batch_1 = ht_2[:3]
outputs_batch_1 = outputs[:3]
print(f' Predictions for the first batch: \n\n{outputs_batch_1}, \
      \n \n Hidden states for the first bactch: \n{hidden_batch_1}')

'''
Predictions for the first batch: 
tensor([[[0.4342, 0.1669, 0.1735, 0.2254],
         [0.2207, 0.2352, 0.3322, 0.2119]],
        [[0.2045, 0.1916, 0.4443, 0.1596],
         [0.4384, 0.1563, 0.1995, 0.2058]],
        [[0.4261, 0.1340, 0.2763, 0.1636],
         [0.1819, 0.1798, 0.4972, 0.1411]]], grad_fn=<SliceBackward>),       
 
 Hidden states for the first bactch: 
tensor([[[ 0.7565, -0.3472],
         [-0.1355, -0.2040]],
        [[-0.1535, -0.5712],
         [ 0.7664, -0.5062]],
        [[ 0.7495, -0.8616],
         [-0.2399, -0.6680]]], grad_fn=<SliceBackward>)
'''
```


<center>Model predictions and hidden states for the first batch</center>

<strong>How well did we perform?</strong>

By looking at the output probabilities we can tell that we did not do so well. However, let’s quantify it using the cross entropy equation! Here we will work our way from the inner term out on the first sequence in the batch. Note, the code will included all 3 sequences in batch 1.



![img](https://cdn-images-1.medium.com/max/1600/1*lUUmNbjMNS1rfX4El9i5VA.png)

**First term:** Element-wise multiplication of the true labels with the log of the predicted labels



![img](https://cdn-images-1.medium.com/max/1600/1*GA9celuV8C1zouFte1eE7Q.png)

<center>Cross entropy term 1 calculation</center>

<strong>Implementation in code:</strong>

```
y[0] * torch.log(outputs_batch_1)

'''
tensor([[[-0.0000, -1.7905, -0.0000, -0.0000],
         [-0.0000, -0.0000, -0.0000, -1.5516]],
        [[-0.0000, -0.0000, -0.8114, -0.0000],
         [-0.0000, -1.8560, -0.0000, -0.0000]],
        [[-0.8532, -0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0000, -0.6987, -0.0000]]], grad_fn=<ThMulBackward>)
'''
```

**Second term:**Summation of remaining values within each sequence. In this step, it is key to note that the axis will be reduced row-wise, only containing the non-zero terms. This will be done in a loop programatically.



![img](https://cdn-images-1.medium.com/max/1600/1*bOLRznERdlJ2tzFcmvy4UQ.png)

<center>Cross entropy term 2 calculation</center>

<strong>Implementation in code:</strong>

```
ce_sums = []
for prediction, label in zip(outputs_batch_1, y[0]):
    ce_sum = torch.sum(label * torch.log(prediction),dim=1)
    ce_sums.append(ce_sum)
ce_sums = torch.stack(ce_sums)
ce_sums

'''
tensor([[-1.7905, -1.5516],
        [-0.8114, -1.8560],
        [-0.8532, -0.6987]], grad_fn=<StackBackward>)
'''
```

**Third term:** Mean of the reduced samples for first sequence within the batch tow-wise. This example calculation was done on the first sequence within batch 1. However, the code implementation covers all 3 sequences in batch 1.


![img](https://cdn-images-1.medium.com/max/1600/1*Xju6WqPDo5SalVFG5Nbk4A.png)
<center>Cross entropy term 3 calculation</center><br>

<strong>Implementation in code:</strong>

```
ce_scores = []
for ce in ce_sums:
    ce = -torch.mean(ce_sums, dim=1)
    ce_scores.append(ce)
ce

'''
tensor([1.6710, 1.3337, 0.7760], grad_fn=<NegBackward>)
'''
```

<strong>Averaging the cross entropy losses of each sequence within batch 1</strong><br>
Note, in practice this step will be done over each mini-batch by keeping a running average of the losses for each batch. It essentially sums up what we calculated for the cross entropy (loss for each sequence in batch 1) and divides it by the number of sequences within the batch.

```
torch.mean(ce)

'''
tensor(1.2602, grad_fn=<MeanBackward1>)
'''
```

<strong>How did we do?</strong><br>
```
def cross_entropy(yhat, y):
    return -torch.mean(torch.sum(y * torch.log(yhat), dim=1))
  
def total_loss(predictions, y_true):
    total_loss = 0.0
    for prediction, label in zip(predictions, y_true):
        cross = cross_entropy(prediction, label)
        total_loss += cross
    return total_loss/ len(predictions)   
  
  # Attached variables 
  params = [Wz, Wr, Wh, Uh, Uz, Ur, bz, br, bh, Wy, by] # iterable of parameters that require gradient computation
  
# Optimizer and training loop
optimizer = torch.optim.SGD(params, lr = 0.01)
max_epochs = 100  # passes through the data
for e in range(max_epochs):
    h = torch.zeros(batch_size, hiddenSize)
    for i in range(num_batches):
        x_in = X[i]
        y_in = y[i]
        
        optimizer.zero_grad() # zero out gradients 
        
        out, h = gru(x_in, h)
        loss = total_loss(out, y_in)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        optimizer.step()
    if e % 10 == 0:
        print(f'Epoch: {e+1}/{max_epochs}')
        print(f'Loss: {loss}')
        print(sample('Ma',10))
              
'''
Epoch: 1/100
Loss: 0.06194562092423439
Matttttttttt
Epoch: 11/100
Loss: 0.06142784655094147
Matttttttttt
Epoch: 21/100
Loss: 0.06091505289077759
MatttttttttM
Epoch: 31/100
Loss: 0.060407400131225586
Matttttttttt
Epoch: 41/100
Loss: 0.05990474298596382
Matttttttttt
Epoch: 51/100
Loss: 0.05940718576312065
Matttttttttt
Epoch: 61/100
Loss: 0.058914750814437866
Matttttttttt
Epoch: 71/100
Loss: 0.05842741206288338
Matttttttttt
Epoch: 81/100
Loss: 0.05794508382678032
Matttttttttt
Epoch: 91/100
Loss: 0.057467926293611526
Matttttttttt
'''
```


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