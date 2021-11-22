# DLCV HW1

For details of the assignment, please refer to the [pdf](https://github.com/kkeen699/DLCV-spring2019/blob/master/hw1/DLCV_hw1.pdf) provided by TAs.

## Problem 1 Bayes Decision Rule

<p align="center"> 
<img src="./image/README/p1.png" alt="drawing" width=""/>
</p>

## Problem 2 Principal Component Analysis
1. Mean face and first four eigenfaces
<p align="center"> 
<img src="./image/README/p2-1.png" alt="drawing" width=""/>
</p>

2. Reconstruction and MSE
<p align="center"> 
<img src="./image/README/p2-2.png" alt="drawing" width=""/>
</p>

3. Apply the k-NN algorithm to classify the testing set.

    First, determin the best k and n values by 3-fold cross-validation.
  
    |        | k = 1 | k = 3 | k = 5 
    ---------|-------|-------|------
    n = 3    | 0.704 | 0.617 | 0.521 
    n = 45   | 0.929 | 0.858 | 0.792
    n = 140  | 0.929 | 0.858 | 0.754    

    Choose k = 1, n = 45, and the recognition rate of the testing set is 0.95625.

## Problem 3 Visual Bag-of-Words