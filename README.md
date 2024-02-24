# Automated Classification of Fruit Spoilage: Detection and Analysis
Elizabeth Bruda, Miles Gordon, Muhamad Imannulhakim, Varsha Jacob, Tejaswini Ramkumar Babu

## Introduction
### Literature Review
The agricultural industry faces a challenge in effectively identifying rotten fruits. Manual classification of fruits is tedious for farmers and prone to human error and fatigue[1]. Unlike humans, machines do not tire after repetitive tasks, making them ideal for this problem. Spoiled fruit poses a risk to fresh produce if not quickly removed. Thus, early detection of rotten fruits is crucial. Computer vision and machine learning can be used to classify fresh and rotten fruits automatically, reducing human effort, cost, and time[2]. 

### Description of Dataset
Our dataset includes fresh and rotten strawberries, peaches, and pomegranates. The images have white backgrounds and are .jpegs. All the images are 300x300. There are 1500 images, with 250 for each of the six classes.

## Problem Definition
### Problem
The goal of our project is to discover and compare methods of machine-learning classification to classify the type of fruit and predict if a fruit is rotten.

### Motivation
We want to help reduce food waste by identifying rotten fruit ahead of time. Unlike humans, machines can quickly and repeatedly perform the same task without getting tired. Previous attempts at solving this problem have used Convolutional Neural Networks (CNN) to extract features from fruit images [1][3]. We plan to refine this approach by integrating CNN as one of many models in sequence, including PCA, SIFT, and Agglomerative Clustering.

## Methods
For preprocessing, we plan to use scaling to normalize image pixel values, data augmentation to generate a more diverse dataset by applying random transformations to existing images (both using tf.keras.preprocessing.image.ImageDataGenerator), and PCA (sklearn.decomposition.PCA) to reduce image dimensionality and lower training computational complexity. We can then extract SIFT descriptors (cv2.SIFT_create) from raw images and perform hierarchical clustering (sklearn.cluster.AgglomerativeClustering) to create a predictive model based on proximity to rotten/fresh clusters, and compare the results to those from training a CNN on our labeled data using categorial cross-entropy loss (tensorflow.keras.Model.fit), which aligns well with a probabilistic multi-class classification model.


## (Potential) Results and Discussion
The quantitative metrics we will use are precision, accuracy, and speed. We’ll use precision since false positives (identifying rotten food as fresh) are dangerous to users. The accuracy rate will measure the model’s correctness. We’ll use time() to get a timestamp before and after calling model.predict() for the model speed. 

We hope to achieve at least a 95% accuracy rate because previous work has achieved this using CNNs [1][3][4]. We hope to have a passing precision score of at least 0.7 and a prediction speed of at most 45 seconds [5] on Google Colab. 

We expect our model will split fruit into 6 classes (each of the 3 kinds of fruits and whether they are rotten). Given an image of a fruit, the model will accurately predict which fruit it is and if it is rotten. We expect that the model may have lower accuracy differentiating between pomegranates and strawberries due to their similar colors. We may encounter overfitting problems since our dataset only has 250 images of each fruit. 


# References
- [1] Mohd Mohsin Ali, M. Raj, and Deepika Vatsa, “FruizNet Using an Efficient Convolutional Neural Network,” Mar. 2023.
- [2] J N V D Tanuia Nerella, Vamsi Krishna Nippulapalli, Srivani Nancharla, Lakshmi Priya Vellanki, and Pallikonda Sarah Suhasini, “Performance Comparison of Deep Learning Techniques for Classification of Fruits as Fresh and Rotten,” Apr. 2023.
- [3] D. V. Dhande and D. D. Patil, “A Deep Learning Based Model for Fruit Grading Using DenseNet,” International Journal of Engineering and Management Research, vol. 12, no. 5, pp. 6–10, Oct. 2022.
- [4] E. Sonwani, U. Bansal, R. Alroobaea, A. M. Baqasah, and M. Hedabou, “An Artificial  Intelligence Approach Toward Food Spoilage Detection and Analysis,” Frontiers in Public Health, vol. 9, Jan. 2022.
- [5] E. Cai, A. Singh, and D. Marculescu, “Learning-based Power and Runtime Modeling for Convolutional Neural Networks,” Carnegie Mellon University, May 2018.


# Gantt Chart
![Team 39 Gantt Chart Screenshot](https://i.imgur.com/DhDmLrW.png "Gantt Chart")
See [here](https://docs.google.com/spreadsheets/d/1m-W8_CN5DLlSRQmso1E5ofZwSR4k7cMV/edit?usp=sharing&ouid=101081220951400011101&rtpof=true&sd=true) for the full chart.

