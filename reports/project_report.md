### Problem Statement
Hierarchical clustering will be implemented to determine plant species that grow in similar geographic regions. Clustering will be conducted on a dataset containing 34,781 unique plant species with corresponding data on states and provinces where each species grows. The dataset includes information on 70 different geographic regions (states/provinces). The ultimate goal is to analyze the plant species of the 5 clusters with greater than 1000 species. There are no privacy concerns as the data is publicly available under the Creative Commons Attribution 4.0 International license and was downloaded from the UC Irvine Machine Learning Repository.

### Data Preprocessing
The initial, uncleaned dataset consisted of comma-delimited strings, where the first value in each string was the plant species and the remaining values were the regions where the species grows. For clustering, the data was converted into a binary matrix, with rows representing plant species and columns representing regions. Each matrix entry indicated whether a given species grows in a given region, with 1 indicating presence and 0 indicating absence.

### Hierarchical Clustering Overview
Clustering is an unsupervised learning technique in which instances in a dataset are partitioned into groups of similar instances, called clusters (GRAD504, 2026b). Hierarchical clustering builds a hierarchy of nested clusters, allowing the number of clusters to be determined after the clustering is completed (GRAD504, 2026c). This is advantageous as it requires limited knowledge of the data and can provide informative visualizations as well as identify uncommonly shaped clusters (GRAD504, 2026c).

**Similary Metric**

Instances are placed into clusters by measuring the distance between instances in the dataset. Various distance functions can be used and offer unique benefits for different types of data (GeeksforGeeks, 2025). For this implementation, Jaccard distance was used as it is designed for binary data where the presence of shared features is more informative than absent features (Scheweinberger, 2026). This aligns with the binary matrix created during data preprocessing, in which each plant species was represented by the regions where it is present or absent. As shown in the equation below, the distance between instances is computed based on features shared by both instances and the features of each individual instance (GRAD504, 2026a). For this implementation, this reflects the regions where both plant species in a pair grow, as well as the regions where each species grows individually.

$d(x_1, x_2) = 1 - \frac{x_1 \cap x_2}{x_1 \cup x_2}$

After pairwise distances are computed, clusters can be merged according to different linkage criteria, including single-link, complete-link, and average-link (GRAD504, 2026c). Single-link uses the nearest pair of points, complete-link uses the farthest pair, and average-link uses the average pairwise distance between points in two clusters (GRAD504, 2026c). Average-link was used as it compares the overall similarity between clusters, provides a balance between single- and complete-linkage. For understanding regions of plant growth, this ensures clusters reflect broader similarity in regional growth patterns rather than being driven by single pairs of species.

**Clustering Algorithm: Agglomerative**

Agglomerative clustering was used in which each instance was first treated as an individual cluster (GRAD504, 2026c). The Jaccard distance between each instance and every other instance in the dataset was then calculated, and the two closest instances were merged into a cluster. This process was repeated until the root of the hierarchy was reached, resulting in a single cluster containing all instances. The function "agglomerative(X)" is a scratch implementation of the clustering algorithm. In each iteration, the algorithm computes distances between instances or clusters across the dataset. Because this process is computationally demanding, applying the scratch implementation to the full dataset of 34,781 plant species was infeasible on a typical workstation. Therefore, a subset of 5,000 species was used for the scratch implementation.

Scipy offers a library for hierarchical clustering that uses more computationally efficient data structures compared to the scratch implementation that relies on many Python loops (SciPy, n.d.-b). Scipy was used to conduct agglomerative hierarchical clustering with average-link and Jaccard distance on the entire dataset. The Scipy implementation results were used for analysis.

### Analysis
The agglomerative hierarchical clustering results were split into 11 flat clusters (Scipy, n.d.-a). 11 clusters were chosen because 5 of those 11 clusters contain greater than 1,000 plant species and will be used for the subsequent analysis and discussion. The table below summarizes the compositions of the 5 clusters with greater than 1,000 species.

| Cluster | Number of species | Number of regions | Top region | % in top | 2nd top region | % in 2nd top |3rd top region | % in 3rd top |
|---|---|---|---|---|---|---|---|---|
| 2 | 3585 | 30 | pr | 97.2 | vi | 40.5 | fl | 28.2 |
| 3 | 1991 | 21 | hi | 99.8 | fl | 2.9 | vi | 0.5 |
| 5 | 1521 | 67 | ak | 61.7 | yt | 41.5 | nt | 37.9 |
| 6 | 17032 | 70 | ca | 53.8 | az | 29.4 | or | 27.9 |
| 9 | 10197 | 70 | nc | 56.9 | ga | 56.6 | va | 54.0 |

It is notable that some clusters are dominated by a single region while others are more geographically diverse. Clusters 2 and 3 are associated with a smaller set of regions, with nearly all species in each cluster growing in the cluster's most common region. By contrast, clusters 5, 6, and 9 represent more than twice as many regions, and only slightly greater than half of the species in each cluster grow in their most common region. This likely reflects differences in species' environmental requirements. Species adapted to unique or specialized climates may have more restricted regional distributions, causing a single region to dominate their clusters. In contrast, species adapted to milder or more broadly suitable conditions may occur across a wider range of regions. When considering the top three regions for each cluster, this trend follows such that clusters 2 and 3 both represent tropical climates with hotter, more extreme weather conditions while clusters 6 and 9 represent subtropical or temperate conditions. Cluster 5 also represents a more extreme climate as its top three regions are in arctic and subarctic North America. Thus, climate seems to be the dominant clustering factor as compared to plant type or use. The given dataset only provides geographic information for each species, so further research would be required to enable specific conclusions to be drawn around plant type and use.

The following table displays species of interest and their corresponding clusters. As shown, all species of interest grow in one of the five largest clusters with six of the nine species growing in cluster 9. As discussed previously, cluster 9 represents a subtropical or temperate environment in which a greater number of species are expected to grow. Cluster 9 appears to represent the southeastern United States and has the most evenly distributed regional composition of the five largest clusters. It therefore follows that many of the species of interest grow in this region. Likewise, clusters 2, 3, and 6 represent more extreme conditions, so it follows that only one of the species of interest is clustered in each of these regions, respectively.

| Species | Cluster |
|---|---:|
| cycas revoluta | 2 |
| huperzia nutans | 3 |
| allium yosemitense | 6 |
| sabal palmetto | 9 |
| rosa gallica | 9 |
| lagerstroemia | 9 |
| syringa | 9 |
| hibiscus syriacus | 9 |
| huperzia lucidula | 9 |

### Reflection

The nature of agglomerative hierarchical clustering likely influenced the clustering results because of the way the algorithm forms clusters. As discussed, each instance begins as its own cluster, and clusters are progressively merged based on their distance from one another (GRAD504, 2026c). Therefore, the choice of distance metric greatly influences the outcomes. In this implementation, the data was preprocessed into a binary matrix where each region was treated as a binary feature. Jaccard distance was selected as the distance metric because the feature set captured presence or absence across regions (Scheweinberger, 2026). If geographic regions were represented using coordinates rather than binary presence/absence features, distance metrics such as Euclidean or Manhattan distance could have been used to measure spatial distance between locations (Scheweinberger, 2026). With spatial data, the results would likely be similar to the binary clustering results because both approaches reflect geography, but using a different distance metric could have produced slightly different clusters.

Furthermore, the choice of average-link affected the clustering results. Average-link merges clusters based on the average pairwise distances between points in the clusters (GRAD504, 2026c). In contrast, single-link merges based on the closest individual pairwise distance between points in two clusters (GRAD504, 2026c). Using single-link may have produced different results by producing long, thin clusters of closely related species (GRAD504, 2026c), potentially creating less compact regional groups. Additionally, complete-link merges based on the farthest individual pairwise distance between points in two clusters (GRAD504, 2026c). As a result, using complete-link may have produced more compact clusters, but it is more sensitive to outliers, which could have impacted the applicability of the outcome (GeeksforGeeks, 2025). Average-link provides a balance between single- and complete-linkage (GeeksforGeeks, 2025), which is why it was chosen for implementation. A greater understanding of the given dataset would aid in determining the optimal linkage metric for the given dataset.

In an alternate scenario focused on plant edibility, the implementation would require an additional data source identifying whether each species is safe for human consumption. The edibility data could likely also be encoded as a binary feature, with 1 representing species considered safe to eat and 0 representing species considered unsafe to eat. Complete-link may be more appropriate for clustering by edibility because it generally creates tighter clusters (GeeksforGeeks, 2025). This would be useful for edibility-focused clustering where edible species should share multiple traits such as edible plant part, preparation requirements, and toxicity risk (Surrette, 2026). While a binary approach would work, it would greatly simplify a complex trait that depends on multiple shared characteristics, but complete-link could partially address this limitation by creating compact and internally consistent clusters.

# References
3.04 - AI as ML: K-Nearest Neighbor (12:11). (2026a). Purdue University. Retrieved March 23, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344456/View. 

5.01: AI as ML: Descriptive Modeling (11:06). (2026b). Purdue University. Retrieved April 9, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344506/View. 

5.05 - Descriptive Modeling: Hierarchical Clustering (14:26). (2026c). Purdue University. Retrieved April 10, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344510/View. 

GeeksforGeeks. (2025, July 12). Types of linkages in hierarchical clustering. https://www.geeksforgeeks.org/machine-learning/ml-types-of-linkages-in-clustering/ 

pandas. (n.d.). pandas.DataFrame. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html 

Python. (2026, April 25). 5. Data Structures. Python documentation. https://docs.python.org/3/tutorial/datastructures.html 

Schweinberger, M. (2026, January 1). Cluster and correspondence analysis in R. LADAL. https://ladal.edu.au/tutorials/cluster_analysis/cluster_analysis.html 

scikit learn. (n.d.). MultiLabelBinarizer. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html 

SciPy. (n.d.-a). Hierarchical clustering (scipy.cluster.hierarchy). https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html 

SciPy. (n.d.-b). Linkage. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html 

Surrette, S. (2026, January 1). Forgotten foods: Introduction to wild edible plants. Mississippi State University Extension. https://extension.msstate.edu/publications/forgotten-foods-introduction-wild-edible-plants 
