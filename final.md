


# Final Report
## Abstract
United States have a higher rate of police killings with 3 people being killed per day on an average. The factors that impact these killings needs to be analysed and studied to understand different aspects leading to the killings. The killings might be impacted by factors like ethnicity, income levels, education etc and also determine the possibility of being an victim of the killings. The analysis include Preprocessing the data by use of techniques like removing outliers, eliminating noise and filling out missing values in an appropriate manner. Then process the data to find the useful information by visualizing it for clear understanding by using histograms, scatter plots etc. Then Data Mining techniques such as Feature Selection, Decision Tree Classification, K-Means Clustering, Hierarchial and Agglomerative Clustering, Correlation Analysis, Nearest Neighbour Classification, and Pricipal Component Analysis will be applied to further analyse the data.
## Experiments/Analysis
**K-Means clustering**:
We want to cluster the data and observe the properties of clusters formed. The important prerequisite before performing K-Means clustering is to find the optimal value of K. Elbow method has been used to find the optimal value of K and the optimal value is found to be **3**. We performed K-Means clustering with K value as 3, the X axis as **h_income** which is found out to be the most important feature in the dataset when feature selection has been performed  and the Y axis as **cause**. We found out that the police killings due to gunshot and taser has been comparatively more in the cities with lower h_income ( household income ) when compared to cities with medium and higher h_income.

![Elbow method](/images/Elbow.PNG)

![K-Means](/images/K-Means.PNG)

**Hierarchical Agglomerative clustering**:

We have used dendrogram to find the optimal number of clusters for performing hierarchical clustering.
Since we found 5 significant branches, we performed hierarchical clustering with **4** clusters and plotted the scatter plot with **cause** on the Y axis and **armed** on the X axis. By observing the scatter plot the following conclusions are drawn:

 - Subject carrying no arms or carrying a knife died only in custody.
 - Subject driving vehicle was not died due to taser. 
 - Subject carrying arms such as firearm, non-lethal firearm and other kinds of        firearm are prone to gunshots, death in custody, taser and struck by vehicle.
 - Subjects disputed with police had not been shot but died either in
   custody, due to taser or struck by vehicle.

![Dendrogram](/images/Dendrogram.PNG)

![Hierarchical](/images/Hierarchical.PNG)

**Decision Tree classification**:

We classified the police killings based on the cause and built a decision tree classifier model to classify police killings with unknown cause. There are more than 20 features in the dataset and all the features may not be significant in the classification. So, we performed feature selection and considered the top 5 features for decision tree classification which are h_income ( household income ), county_income ( county level median household income ), p_income ( personal income ), pop ( population ) and pov ( poverty rate ) . We split the testing and training data in the ratio of 20:80. We have built the decision tree with depths ranging from 2 to 5. The testing accuracy is found to be varying between 80 and 85 most of the times.

![Decision Tree](/images/Decision%20Tree.png)

**Principal Component Analysis**:

when the dataset contains a large number of features, it will affect the accuracy of the training model and the time it takes to train. To solve this problem, we can identify important features that give maximum variance and this can be done by Principal Component Analysis.if we have a big dataset or want to do data compression PCA is the best option. Here we aim to find important features using PCA.

Steps to perform PCA

1.In our dataset, we took an age, year, pop, p_income, h_income,county_income, comp_income, pov, urate, college features to perform PCA as these features are numeric and took cause as target column.
2.standardize the available data and calculate the covariance matrix.
3.do Eigen-decomposition on the matrix and sort the Eigenvalues in decreasing order
4.multiply original normalized data by eigenvector of the covariance matrix which has the highest eigenvalue.
5. the resultant reduced PCA maximizes the original data variance.

After performing PCA, we projected the original data into two dimensional PCA space.
[0.48854956 0.13008248] are the variance ratio for PC1 and PC2. as PC1 has the highest variance ratio we will consider PC1. considering the values of the features in pc1 we selected top 5 features to perform K-Nearest Neighbour classification. top 5 features are age, p_income ,h_income, pov, comp_income.
 
![pca](/images/pca.png)

**K-Nearest Neighbour classification**:

it this algorithm we store all available data and classify the new data based on the similarity. if we need to classify a new data it calculates k -nearest neighbours and go with the majority group.k plays an important role in deciding the model's accuracy.there is no specific pre-defined method to calculate k we have to find out based on trial and error method. here, we have taken important features by performing PCA and they are age, p_income,h_income, pov, comp_income.For training the model, divide the data into test and train data in the ration 1:4. To find the k, we have run the knn method from k value in the range of(1,15).we got good and stable accuracy at k=6. so we considered k value as 6 and performed Knn classification. we calculate the similarity based on the distance between two points and mostly we use Euclidean distance. finally, we got the model accuracy score of 91 percentage.

![k](/images/k.png)

![cm](/images/cm.png)

## Comparisons

While performing clustering using elbow method to find the number of clusters give a very limited and precise scope for finding the number of clusters whereas using dendrogram for the same purpose give a vast range when compared to elbow method to come up with the number of clusters.

The nearest neighbor classifier model is able to perform well in predicting the cause of the police killing with predictive accuracy more than 90% most of the times when compared to decision tree classifier model which could yield a predictive accuracy ranging from 80% and 85% most of the time.

## Conclusion
We have built the Classification Models to classify the cause of police killings to arrive at the cause of killing based on different features and also determine the cause of killing when the cause is unknown. We found out that the police killings due to gunshot and taser has been comparatively more in the cities with lower h_income ( household income ) when compared to cities with medium and higher h_income. Subjects carrying no arms or carrying a knife died only in custody whereas subjects carrying arms are prone to gunshots, death in custody, taser and struck by vehicle but the subjects disputed with police died either in custody, due to taser or struck by vehicle. K-Nearest Neighbour classifier model gave us the accuracy score of 0.91 percentage whereas Decision Tree Classifier model gave an accuracy ranging between 0.8 and 0.85 

## Steps to run the code:
**Google Colaboratory**:
 - Open Google Colaboratory.
 - Click NEW NOTEBOOK.
 - Copy the main.py file and paste in the notebook.
 - Import the police_killings.csv file into Google Colaboratory and mention the path in the line "data = load_data('police_killings.csv')" in place of police_killings.csv.
 - The above line of code can be found in main function.
 - Hit the Run cell button.
 
 **Pycharm**:
 
 - Open Pycharm editor.
 - Import the main.py file and police_killings.csv file .
 - Install sklearn, matplotlib, scipy, pandas and numpy.
 - Open the terminal and navigate to the path where main.py is present and execute the command ' pip3 install plotly '.
 - Run the main.py file.
 - All the visualizations appear automatically and the images of decision trees created gets saved in the folder where main.py is present.   
