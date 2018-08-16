## DomClick
Text classification baseline in jupyter-notebooks

### Files description:
* **explore_dataset.ipynb** - general overview of a dataset including the length of the sentences
* **embedding.ipynb** - includes text tokenization, removing outliers (frequent and unique words), Bag Of Words (BOW) and Term Frequency - Inverse Document Frequency (TF-IDF) embedding
* **NaiveBayes.ipynb** - implements a Naive Bayes probabilistic model to the BOW representation of a dataset
* **CNN_LSTM_classifier.ipynb** - implements a Deep Learning model using the CNN and Bidirectional LSTM on the same BOW embedding. Uses only sentences with length of 300 or less (~93% of a dataset)
* **decomposition.ipynb** - performs matrix factorization (PCA and SVD) on the any embedding (BOW or TF-IDF). Also implements Fisher Vector features engineering by using the GMM model (code is borrowed from reference below)
* **downsampling_and_classification.ipynb** - implements the downsampling of the decomposed dataset, data normalization and classification with Boosting, Bagging and etc.

### Order
1. explore_dataset
2. embedding
	* 3a. NaiveBayes
	* 3b. CNN_LSTM_classifier
	* 3c. decomposition ---> downsampling_and_classification

After **embedding** the flow goes in three different approaches:
* Probabilistic model - **NaiveBayes** 
* Deep Learning - **CNN_LSTM_classifier** 
* **Decomposition**, **downsampling_and_classification** - includes SVD, PCA decomposition, GMM model for feature engineering, balancing the dataset and classification using gradient boosting, bagging, logistic regression, linear discriminant analysis and MLP.

### To install the environment
`conda config --append channels conda-forge`
`conda create -n "name_of_new_environment" --file package-list.txt`

### Conclusion
Best performed baseline is:
TF-IDF encoding (with filtering of outliers) -> SVD decomposition (n_components=100) -> GMM feature extractor (n_gaussians=2) -> Downsample the majority class by clusters centroids method -> XGBoost classification (without the data normalization)

### Links used
* http://scikit-learn.org/stable/
* https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
* https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d

### References
* https://gist.github.com/danoneata/9927923

### Python version
Python 3.6.3

