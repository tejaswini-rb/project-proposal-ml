# Description:

This repository explores data preprocessing, hierachical clustering, decision trees, and CNN architectures. for classifying fresh and rotten fruit. The fruits in our dataset are peaches, strawberries, and pomegranates.

- `dir`: Our README directory in .txt form.

- `CNNs`: This folder contains the vanilla CNN model, EfficientNet, ResNet50, and MobileNetV2 models.  Image classification using TensorFlow and Keras; we did data preparation with train-validation-test split, exploratory data visualization, data augmentation, CNN model and training, and performance evaluation with accuracy and loss metrics.
    - `CNN_10_Epochs.ipynb`: This file has the results for a 10 epoch run and a default learning rate (Adam optimizer).
    - `CNN_30_Epochs.ipynb`:  This file has the results for a 30 epoch run and a learning rate of 0.006 (Adam optimizer), along with Top K and speed metrics.
    - `CNN_50_Epochs.ipynb`: This file has the results for a 50 epoch run and a learning rate of 0.006 (Adam optimizer), along with Top K and speed metrics.
    - `CNN_ResNet_MobileNetV2_EfficientNet.ipynb`: This file explores the accuracy, top K, and speed of Resnet50, EfficientNet, and MobileNetV2.

- `DecisionTree.ipynb`: Juypter notebook containing the code for training a decision tree classifier on the images in the training dataset and performing predictions on the images in the test dataset. These images have already been preprocessed.

- `Hierarchical_Clustering.ipynb`: Juypter notebook extracting features using the SIFT method and does hierarchical clustering to categorize these images into groups based on visual similarity.

- `Hierarchical_Clustering_Color_Histograms.ipynb` Juypter notebook with histograms for hierarchical clustering.

- `Data_Preprocessing.ipynb`: Juypter notebook for splitting the data into train, test, and validation sets as well was performing image preprocessing such as rotation, brightness adjustment, shear, zoom, and flipping

- `fruit_dataset`: Directory with our fruit dataset of fresh and rotten, peaches, strawberries, and pomegranates.
- `fruit_data.zip`: Directory with preprocessed images.
- `requirements.txt`: File with the requirements, packages, and dependencies.
