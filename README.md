# MachineLearning
Book Recommendation System using Collaborative Filtering
# <span style="color:Green">Building a Book Recommendation System</span>

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm
style.use('seaborn')
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

### Importing the .csv Files

Book_Data = pd.read_csv("BX-Books.csv",sep=';',encoding="latin-1", error_bad_lines=False)
Book_Ratings_Data = pd.read_csv("BX-Book-Ratings.csv",sep=';',encoding="latin-1", error_bad_lines=False)
User_Data = pd.read_csv("BX-Users.csv",sep=';', error_bad_lines=False, encoding="latin-1")

# <span style="color:Purple">Exploratory Data Analysis</span>

## Data Preparation and Data Cleaning

#shows count of missing values in each column of Book_Data df
Book_Data.isna().sum() 

#shows count of missing values in each column of Book_Ratings_Data df

Book_Ratings_Data.isna().sum() 

#shows count of missing values in each column of User_Data df

User_Data.isna().sum() 

#head is used to get first 5 rows

Book_Data.head() 

# Renaming Book_Data df column names

Book_Data.rename(columns = {"Book-Title" : "Book_Title","Book-Author": "Book_Author","Year-Of-Publication":"Year_Of_Publication","Image-URL-S":"Image_URL_S","Image-URL-M":"Image_URL_M","Image-URL-L": "Image_L",}, inplace=True)


#populates first 5 rows from Book_data df

Book_Data.head() 

# Renaming column names from Book_Ratings_Data

Book_Ratings_Data.rename(columns = {"Book-Rating":"Book_Rating","User-ID":"User_ID"},inplace = True)


Book_Ratings_Data

User_Data.rename(columns = {"User-ID":"User_ID"},inplace = True) #renaming User_Data column name

User_Data.head()

In the User File, The Location column is represented with state and country. To which we are splitting the Country and created a new column as Country. 

import re
countries=[]
for i in list(User_Data["Location"]):  
    chunks = i.split(", ")        #splitting location column data by country wise
#     print(i)
#     chunks = re.split('[_,][_,]',i)
    if chunks[-1] != ',':
        countries.append(chunks[-1])
    else:
        countries.append(None)
countries

User_Data["Country"] = countries

User_Data.head(5)

User_Data["Country"].fillna("usa", inplace = True) #replaces null values with usa

### Top 10 Authors with most selling Books

plt.figure(figsize=(8,8))
ax = sns.barplot(top_15_authors['Book_Title'], top_15_authors.index, palette='CMRmap_r')

ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
plt.show()

The above bar graph shows Book authors with total number of books published. Agatha Christie is the top most author with most number of books published.

### Countries with Total counts of Users rated the Books

cm=sns.light_palette('red',as_cmap=True)
popular=User_Data.Country.value_counts().to_frame()[:10]
popular.rename(columns={'Country':'Count_Users_Country'},inplace=True)
popular.style.background_gradient(cmap=cm)

The above table shows number of users per country. We can see that first highest country to have maximum number of users is usa.

### Understanding the Distributions in Age Column

sns.distplot(User_Data.Age)

Seaborn distplot showing age distribution. We can observe that the density is more across age group 15 to 45.

f,ax=plt.subplots(1,2,figsize=(12,7))
#box-plot showing outliers in age and book rating columns.
sns.boxplot(y='Book_Rating', data=Book_Ratings_Data,ax=ax[0])
ax[0].set_title('Outlier data in Rating Book column')
sns.boxplot(y='Age', data=User_Data,ax=ax[1])
ax[1].set_title('Outlier data in Age column')

Two box-plots representing outlier data in Book rating and Age column. Here, age column has more number of outliers.

sorted(Book_Ratings_Data['Book_Rating'].unique()) 

#converting all the outlier age data to nan
User_Data.loc[(User_Data.Age > 100 ) | (User_Data.Age < 5),'Age']=np.nan 

User_Data.Age.plot.hist(bins=20,edgecolor='black',color='Yellow')

Box-plot representing age data after outlier removal. We removed unnecessary data like age greater than 100 and less than 5.

#checking skewness for age
round(User_Data.Age.skew(axis=0,skipna=True),3)

# Series of users data live in which country 
countryUsers = User_Data.Country.value_counts()

country=countryUsers[countryUsers>=5].index.tolist() 

#range of age users in country registered and had participation 
RangeOfAge = User_Data.loc[User_Data.Country.isin(country)][['Country','Age']].groupby('Country').agg(np.mean).to_dict()

for k,v in RangeOfAge['Age'].items():
    User_Data.loc[(User_Data.Age.isnull())&(User_Data.Country== k),'Age'] = v

#returns number of missing values in the dataset
User_Data.isnull().sum() 

#filling na value from median
medianAge = int(User_Data.Age.median())
User_Data.loc[User_Data.Age.isnull(),'Age']=medianAge

#returns number of missing values in the dataset
User_Data.isnull().sum()

### Data cleaning for Book Author and Year of Pubication Column

#Book_Author column that has nan value
Book_Data[Book_Data.Book_Author.isnull()]

Book_Data.loc[(Book_Data.ISBN=='9627982032'),'Book_Author']='other'
Book_Data.loc[(Book_Data.ISBN=='193169656X'),'Publisher']='other'
Book_Data.loc[(Book_Data.ISBN=='1931696993'),'Publisher']='other'

Book_Data.loc[Book_Data.ISBN=='2070426769','Year_Of_Publication']=2003
Book_Data.loc[Book_Data.ISBN=='2070426769','Book_Author']='Gallimard'

Book_Data.loc[Book_Data.ISBN=='0789466953','Year_Of_Publication']=2000
Book_Data.loc[Book_Data.ISBN=='0789466953','Book_Author']='DK Publishing Inc'
Book_Data.loc[Book_Data.ISBN=='078946697X','Year_Of_Publication']=2000
Book_Data.loc[Book_Data.ISBN=='078946697X','Book_Author']='DK Publishing Inc'

### Understanding the Distributions in Year Of Publication Column

Book_Data.Year_Of_Publication=Book_Data.Year_Of_Publication.astype(np.int32)

#summerizing data for Book_Data df
Book_Data.info() 

sns.distplot(Book_Data['Year_Of_Publication'])

From the seaborn histplot above, year of publication in the year 2000 is showing highest peak.

Books= Book_Data.copy()

Books = Books[(Books['Year_Of_Publication']>=1950) & (Books['Year_Of_Publication']<=2016)]
sns.distplot(Books['Year_Of_Publication'])

 Generated distplot that shows line curve of books published from year 1950 to 2016.The graph peaked at 2001.

print(sorted(Book_Data.Year_Of_Publication.unique()))

#Putting NAN value to books published after 2021 and 0 since it's not normal.
Book_Data.loc[(Book_Data.Year_Of_Publication>=2021)|(Book_Data.Year_Of_Publication==0),'Year_Of_Publication']=np.NAN

Book_Data.isnull().sum()

#Replacing the year of publication null values with mean
Book_Data.loc[Book_Data.Year_Of_Publication.isnull(),'Year_Of_Publication'] = round(Book_Data.Year_Of_Publication.mean())

author=Book_Data[Book_Data.Year_Of_Publication.isnull()].Book_Author.unique().tolist()

RangeYearOfPublication = Book_Data.loc[Book_Data.Book_Author.isin(author)][['Book_Author','Year_Of_Publication']].groupby('Book_Author').agg(np.mean).round(0).to_dict()

### Understanding the Ratings Distribution

#creating a new Rating_book dataset
ratings_new = Book_Ratings_Data[Book_Ratings_Data.ISBN.isin(Book_Data.ISBN)]
ratings_new = ratings_new[ratings_new.User_ID.isin(User_Data.User_ID)]

ratings_0 = ratings_new[ratings_new.Book_Rating ==0]
ratings_1to10 = ratings_new[ratings_new.Book_Rating !=0]
# Create column Rating average 
ratings_1to10['rating_Avg']=ratings_1to10.groupby('ISBN')['Book_Rating'].transform('mean')
# Create column Rating sum
ratings_1to10['rating_sum']=ratings_1to10.groupby('ISBN')['Book_Rating'].transform('sum')

#merging datasets
dataset=User_Data.copy()
dataset=pd.merge(dataset,ratings_1to10,on='User_ID')
dataset=pd.merge(dataset,Book_Data,on='ISBN')

fig, ax = plt.subplots(figsize=(9,6))
sns.countplot(data=ratings_1to10,x='Book_Rating',ax=ax)

The above generated countplot/barplot shows the users count for Book_Rating. Majority of the users gave 8 for book_rating and second highest users count was given for rating 10.


## <span style="color:Purple">Popularity Based Approach</span>

dataset.shape

print(dataset.columns.tolist())

dataset.head()

dataset['Count_All_Rate']=dataset.groupby('ISBN')['User_ID'].transform('count')

C = dataset['rating_Avg'].mean()
m = dataset['Count_All_Rate'].quantile(0.90)
Top_books = dataset.loc[dataset['rating_sum']>= m]
print(f'C={C}, m={m}')
Top_books.shape

def weighted_rating(x, m=m, C=C):
    v = x['Count_All_Rate']
    R = x['rating_Avg']
    return (v/(v+m) * R) + (m/(m+v) * C)
Top_books['Score'] = Top_books.apply(weighted_rating, axis=1)

Top_books = Top_books.sort_values('Score', ascending = False)

# cm=sns.light_palette('yellow',as_cmap=True)
# count all rate means include users rated 0 to book
popular=dataset.groupby(['Book_Title','Count_All_Rate','rating_Avg','rating_sum']).size().reset_index().sort_values(['rating_sum','rating_Avg',0],
                                                                                                            ascending=[False,False,True])[:20]
popular.rename(columns={0:'Score'},inplace=True)
popular.style.background_gradient(cmap=cm)

## <span style="color:Purple">Collaborative Filtering Using k-Nearest Neighbors (kNN)</span>

#ratings has 1031136 rows and three columns

ratings_new.shape

#excluding books with less than 100 ratings and users with less than 50 ratings

counts1 = ratings_1to10['User_ID'].value_counts()
ratings = ratings_1to10[ratings_1to10['User_ID'].isin(counts1[counts1 >= 100].index)]
counts = ratings['Book_Rating'].value_counts()
ratings = ratings[ratings['Book_Rating'].isin(counts[counts >= 50].index)]

#merging ratings and book df on ISBN

combine_book_rating = pd.merge(ratings, Books, on='ISBN')
columns = ['Year_Of_Publication', 'Publisher', 'Book_Author', 'Image_URL_M', 'Image_URL_S', 'Image_L']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()

#grouping by book_title and creating a new column total_rating_count

combine_book_rating = combine_book_rating.dropna(axis=0, subset = ['Book_Title'])

book_ratingCount = (combine_book_rating.
                    groupby(by=['Book_Title'])['Book_Rating'].count().reset_index().
                    rename(columns = {'Book_Rating': 'Total_Rating_count'})
                    [['Book_Title','Total_Rating_count']]
                   )

book_ratingCount.head(10)

#combining rating data with total_rating_count

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'Book_Title', right_on = 'Book_Title', how = 'left')
rating_with_totalRatingCount.head()

rating_with_totalRatingCount.shape

#creating a popularity_threshold and setting it to 20.

popularity_threshold = 20
rating_popular_book = rating_with_totalRatingCount.query('Total_Rating_count >= @popularity_threshold')
rating_popular_book.head()

rating_popular_book.shape

#dropping duplicates and assigning variables to generate pivot table

from scipy.sparse import csr_matrix
rating_popular_book = rating_popular_book.drop_duplicates(['User_ID', 'Book_Title'])
rating_popular_book_pivot = rating_popular_book.pivot(index = 'Book_Title', columns = 'User_ID', values = 'Book_Rating').fillna(0)
rating_popular_book_matrix = csr_matrix(rating_popular_book_pivot.values)

### Creating a User - Book Matrix

rating_popular_book_pivot.head()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm= 'brute')
model_knn.fit(rating_popular_book_matrix)

#creating a query_index variable to generate random index number

query_index = np.random.choice(rating_popular_book_pivot.shape[0])
print(query_index)

#applying model and setting n_neighbours to 6.

distances,indices = model_knn.kneighbors(rating_popular_book_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)

#Printing book_title for index 6

rating_popular_book_pivot.index[query_index]

#Using distances to create 5 recommendations based on index number generated

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("recommendations for {0}\n".format(rating_popular_book_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i,rating_popular_book_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


## <span style="color:Purple">KMeans with Principal Component Analysis</span>

ratings_new.shape

# Taking the mean of rating given by each user
User_rating_mean = ratings_new.groupby('User_ID')['Book_Rating'].mean()
user_rating = ratings_new.set_index('User_ID')
user_rating['mean_rating'] = User_rating_mean
user_rating.reset_index(inplace=True)
# Keeping the books in which users "likes" the book
user_rating = user_rating[user_rating['Book_Rating'] > user_rating['mean_rating']]
# Initializing a dummy variable for future use
user_rating['is_fav'] = 1
print(user_rating.shape)
user_rating.head()

# Keeping the users who like more than 10 books and less than 100 books 

val = user_rating['User_ID'].value_counts()
list_to_keep = list(val[(val>10) & (val<100)].index)
user_rating = user_rating[user_rating['User_ID'].isin(list_to_keep)]
user_rating.shape

### i) Creating a Matrix for User and Item

Book_UserMatrix = pd.pivot_table(user_rating,index='User_ID',columns='ISBN',values='is_fav')
Book_UserMatrix.fillna(value=0,inplace=True)
print(Book_UserMatrix.shape)


Book_UserMatrix.head(10)

### ii) Principal Component Analysis for Dimension Reduction

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

#applying pca fit and tranform to user matrix
pca.fit(Book_UserMatrix)
pca_fit = pca.transform(Book_UserMatrix)

pca_fit = pd.DataFrame(pca_fit,index=Book_UserMatrix.index)
pca_fit

## K-means Clustering

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#Generating scatter plot for 3 clusters
KMeansModel = KMeans(n_clusters=3)
plt.rcParams['figure.figsize'] = (6, 6)
clusters = KMeansModel.fit_predict(pca_fit)
cmhot = plt.get_cmap('brg')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca_fit[0], pca_fit[2], pca_fit[1],c=clusters,cmap=cmhot)
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()

From the above scatter plot, we can observe that the green cluster is closely packed in the center whereas the blue and the red are spread out.

### a) Elbow Method

TSS = []
for i in range(2,26):
    ElbowModel = KMeans(n_clusters=i,random_state=0)
    #using elbow method to find the optimal k value
    ElbowModel.fit(pca_fit)
    TSS.append(ElbowModel.inertia_)
plt.plot(range(2,26),TSS,'-')

The elbow model generated here has an elbow point at k = 3 and k = 6

### b) Silhouette Analysis

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

for n in [3,4,5,6,7,8]:
    ax1 = plt.figure().gca()
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(pca_fit) + (n + 1) * 10])
    km = KMeans(n_clusters=n,random_state=0)
    clusters = km.fit_predict(pca_fit)
    silhouette_avg = silhouette_score(pca_fit, clusters)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_values = silhouette_samples(pca_fit, clusters)
    y_start = 10
    for i in range(n):
        ith_cluster = np.sort(silhouette_values[clusters==i])
        cluster_size = ith_cluster.shape[0]
        y_end = y_start + cluster_size 
        ax1.fill_betweenx(np.arange(y_start, y_end),
                          0, ith_cluster)
        ax1.text(-0.05, y_start + 0.5 * cluster_size, str(i))
        y_start = y_end + 10

Above graphs show silhouette analysis for KMeans clustering on data with n_clusters = 3,4,5,6,7,8. We can see the best cluster at k=4.

Kmeans_final = KMeans(n_clusters=4,random_state=0).fit(pca_fit)
Book_UserMatrix['cluster'] = Kmeans_final.labels_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca_fit[0], pca_fit[2], pca_fit[1],c=Book_UserMatrix['cluster'],cmap=cmhot)
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()
# Gettings the books for each cluster
cl1_books = Book_UserMatrix[Book_UserMatrix.cluster == 0].mean()
cl2_books = Book_UserMatrix[Book_UserMatrix.cluster == 1].mean()
cl3_books = Book_UserMatrix[Book_UserMatrix.cluster == 2].mean()
cl4_books = Book_UserMatrix[Book_UserMatrix.cluster == 3].mean()
# Getting the users for each cluster
cl1_users = Book_UserMatrix[Book_UserMatrix.cluster == 0].index
cl2_users = Book_UserMatrix[Book_UserMatrix.cluster == 1].index
cl3_users = Book_UserMatrix[Book_UserMatrix.cluster == 2].index
cl4_users = Book_UserMatrix[Book_UserMatrix.cluster == 3].index

The above scatter plot consists of 4 clusters that represent books and users. Blue cluster is in the middle with tightly packed centroid wheres clusters in green,purple and brown are spread across.

### Cluster 1 (Predicting top 5 books and top 5 authors)

def cluster_books_des(Ser):
    bks = pd.DataFrame(Ser).merge(Book_Data,left_index=True,right_on='ISBN',how='left')
    bks.rename(columns={0:'avg_score'},inplace=True)
    bks.sort_values(by='avg_score',ascending=False,inplace=True)
    print('Median Year:',int(bks['Year_Of_Publication'].median()))
    print('\nTop 5 Books\n')
    Top5_books = bks.index[:5]
    for i,isbn in enumerate(Top5_books):
        print(str(i+1)+'.',bks.loc[isbn]['Book_Title'])
    Top5_authors = bks['Book_Author'].unique()[:5]
    print('Top 5 Authors\n')
    for i,auth in enumerate(Top5_authors):
        print(str(i+1)+'.',auth)
cluster_books_des(cl1_books)

def cluster_user_des(Ser):
    cl_user = User_Data[User_Data['User_ID'].isin(list(Ser))]
    print('Most Common Location:',cl_user['Location'].mode()[0])
    print('\nMean Age:',cl_user['Age'].mean())
    sns.distplot(cl_user['Age'])
    plt.yticks([])
cluster_user_des(cl1_users)

From the above histplot, we can conclude that the cluster 1 has predicted age group 35 to 40 has the maximum density.

### Cluster 2 (Predicting top 5 books and top 5 authors)

cluster_books_des(cl2_books.drop('cluster'))

cluster_user_des(cl2_users)

From the above histplot, we can conclude that the cluster 2 has predicted age group 35 to 40 has the maximum density.

### Cluster 3 (Predicting top 5 books and top 5 authors)

cluster_books_des(cl3_books.drop('cluster'))

cluster_user_des(cl3_users)

From the above histplot, we can conclude that the cluster 3 has predicted age group 38 to 40 has the maximum density.

### Cluster 4 (Predicting top 5 books and top 5 authors)

cluster_books_des(cl4_books.drop('cluster'))

cluster_user_des(cl4_users)

From the above histplot, we can conclude that the cluster 4 has predicted age group 25 to 30 has the maximum density.

## <span style="color:Purple">Collaborative Filtering using SURPRISE</span>

Surprise Library : 

The Surprise package in Python provided all the tools we needed to test out multiple algorithms for Collaborative Filtering and then guided me through tuning the parameters and cross validating to determine the optimal model. In order to this, we chose three versions of the data to analyze. First, we looked at the most popular books only, filtering down to those with at least 10 book ratings and at least 30 user ratings. We then created a list of midlist books by filtering down to 2 book rating and 20 user ratings. Finally, We used the full list to include books that have as little as 1 book rating and 1 user rating.

conda install -c conda-forge scikit-surprise

from surprise import Reader
from surprise import Dataset
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import accuracy
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV


combine_book_rating

# full df, even books with very few ratings or users

BookRating_full = combine_book_rating 
reader = Reader(rating_scale=(1, 10))
Data_full = Dataset.load_from_df(BookRating_full[['User_ID', 'ISBN', 'Book_Rating']], reader)

benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(),KNNBaseline(), KNNWithMeans(), KNNWithZScore(), BaselineOnly()]:
    # Perform cross validation
    results = cross_validate(algorithm, Data_full, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
print("+++ Algorithm results for full data +++") 
surprise_results # lowest = SVDpp

# trim df (2/book, 20/user = RS for backlist books)

min_book_ratings = 2
filter_books = combine_book_rating['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 20
filter_users = combine_book_rating['User_ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_mid = combine_book_rating[(combine_book_rating['ISBN'].isin(filter_books)) & (combine_book_rating['User_ID'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(combine_book_rating.shape))
print('The midlist data frame shape:\t{}'.format(df_mid.shape))

data2 = Dataset.load_from_df(df_mid[['User_ID', 'ISBN', 'Book_Rating']], reader)

benchmark2 = []
# Iterate over all algorithms
for algorithm in [SVD(),KNNBaseline(), KNNWithMeans(), KNNWithZScore(), BaselineOnly()]:
    # Perform cross validation
    results2 = cross_validate(algorithm, data2, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp2 = pd.DataFrame.from_dict(results2).mean(axis=0)
    tmp2 = tmp2.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark2.append(tmp2)

surprise_results2 = pd.DataFrame(benchmark2).set_index('Algorithm').sort_values('test_rmse')
print("+++ Algorithm results for midlist data +++") 
surprise_results2 # lowest = SVDpp

# trim df (10/book, 30/user = RS for most popular books)

min_book_ratings = 10
filter_books = combine_book_rating['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 30
filter_users = combine_book_rating['User_ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_pop = combine_book_rating[(combine_book_rating['ISBN'].isin(filter_books)) & (combine_book_rating['User_ID'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(combine_book_rating.shape))
print('The midlist data frame shape:\t{}'.format(df_pop.shape))

data3 = Dataset.load_from_df(df_pop[['User_ID', 'ISBN', 'Book_Rating']], reader)

benchmark3 = []
# Iterate over all algorithms
for algorithm in [SVD(),KNNBaseline(), KNNWithMeans(), KNNWithZScore(), BaselineOnly()]:
    # Perform cross validation
    results2 = cross_validate(algorithm, data3, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp2 = pd.DataFrame.from_dict(results2).mean(axis=0)
    tmp2 = tmp2.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark3.append(tmp2)

surprise_results2 = pd.DataFrame(benchmark3).set_index('Algorithm').sort_values('test_rmse')
print("+++ Algorithm results for midlist data +++") 
surprise_results2 # lowest = SVDpp

Using the top rated algorithms above, I choose to run a GridSearchCV on KNNBaseline, BaselineOnly, and KNNWithMeans. After completing the gridsearches, I ran 10-fold cross validation on each of the tuned models and plotted the results.Based on surprise results, KNNBaseline ultimely performed best compared to BaselineOnly,KNNWithMeans and KNNWithZScore.

## Grid Search on Top algorithms to identify optimal parameters

### Top algorithms = KNNBaseline, BaselineOnly, KNNWithMeans

GridData = Dataset.load_from_df(combine_book_rating[['User_ID', 'ISBN', 'Book_Rating']], reader)

###   a) GridSearch on KNNBaseline

# gridsearch on KNNBaseline
pg_KNNBaseline = {'bsl_options': {'method': ['als']},
              'k': [10, 30, 50],
              'sim_options': {'name': ['msd', 'cosine', 'pearson', 'pearson_baseline'],
                              'min_support': [1, 5],
                              'user_based': [True]}
              }
gs_KNNBaseline = GridSearchCV(KNNBaseline, pg_KNNBaseline, measures = ['rmse'], cv = 3)
gs_KNNBaseline.fit(GridData)

print(gs_KNNBaseline.best_score['rmse'])

print(gs_KNNBaseline.best_params['rmse'])

### b) GridSearch on BaselineOnly

# gridsearch on BaselineOnly
pg_BaselineOnly = {'bsl_options': {'method': ['als', 'sgd'],
                                  'n_epochs': [10, 25, 50]}
              }
gs_BaselineOnly = GridSearchCV(BaselineOnly, pg_BaselineOnly, measures = ['rmse'], cv = 3)
gs_BaselineOnly.fit(GridData)

print(gs_BaselineOnly.best_score['rmse'])

print(gs_BaselineOnly.best_params['rmse'])

### c) GridSearch on KNNWithMeans

# gridsearch on KNNWithMeans
pg_KNNWithMeans = {'name': ['cosine', 'msd', 'pearson', 'pearson_baseline'],
                   'user_based': [True, False]
              }
gs_KNNWithMeans = GridSearchCV(KNNWithMeans, pg_KNNWithMeans, measures = ['rmse'], cv = 3)
gs_KNNWithMeans.fit(GridData)

print(gs_KNNWithMeans.best_score['rmse'])

print(gs_KNNWithMeans.best_params['rmse'])

## Performing Cross Validation on Top algorithms and Plot their RMSE

import surprise
from sklearn.model_selection import train_test_split

kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into 10 folds

rawTrain,rawholdout = train_test_split(Book_Ratings_Data, test_size=0.25 )
# when importing from a DF, you only need to specify the scale of the ratings.
reader = surprise.Reader(rating_scale=(1,10)) 
#into surprise:
Ratingdata = surprise.Dataset.load_from_df(Book_Ratings_Data, reader)
# holdout = surprise.Dataset.load_from_df(rawholdout,reader)

rmseKNNBaseline = []
rmseBaselineOnly = []
rmseKNNWithMeans = []

# KNNBaseline algorithm

sim_options = {'name': 'pearson_baseline','user_based': True}
bsl_options = {'method': 'als'}

algo_KNNBaseline = surprise.prediction_algorithms.knns.KNNBaseline(k = 30, sim_options = sim_options, 
                                                                   bsl_options = bsl_options, verbose=True)
for trainset, testset in kSplit.split(GridData):
    algo_KNNBaseline.fit(trainset)
    predictionsKNN = algo_KNNBaseline.test(testset)
    rmseKNNBaseline.append(surprise.accuracy.rmse(predictionsKNN,verbose=True))

# BaselineOnly algorithm

pg_BaselineOnly = {'bsl_options': {'method': ['sgd'],
                                  'n_epochs': [25]}}
algo_BaselineOnly = surprise.prediction_algorithms.baseline_only.BaselineOnly(pg_BaselineOnly)
for trainset, testset in kSplit.split(GridData):
    algo_BaselineOnly.fit(trainset)
    predictionsBaselineOnly = algo_BaselineOnly.test(testset)
    rmseBaselineOnly.append(surprise.accuracy.rmse(predictionsBaselineOnly,verbose=True))

# KNNWithMeans algorithm

pg_KNNWithMeans = {'name': ['cosine'],
                  'user_based': True}
algo_KNNWithMeans = surprise.prediction_algorithms.knns.KNNWithMeans()
for trainset, testset in kSplit.split(GridData): 
    algo_KNNWithMeans.fit(trainset)
    predictionsKNNWithMeans = algo_KNNWithMeans.test(testset)
    rmseKNNWithMeans.append(surprise.accuracy.rmse(predictionsKNNWithMeans,verbose=True))

#plt.plot(rmseSVDpp,label='SVDpp')
plt.plot(rmseKNNBaseline,label='KNNBaseline')
plt.plot(rmseBaselineOnly,label='BaselineOnly')
plt.plot(rmseKNNWithMeans,label='KNNWithMeans')

plt.xlabel('folds')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

From the graph above we can observe that the accuracy is highest for KNNWithMeans at 1.78. Whereas, the accuracy for KNNBaseline and BaselineOnly is 1.58 and 1.48 repectively.

## Using one Optimized model to recommend books - KNN Baseline Approach 

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

from surprise import SVDpp
from surprise import Dataset


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Train the optimized algorithm on the dataset

trainset = GridData.build_full_trainset()
algo_KNNBaseline.fit(trainset)

# predict ratings for all user/item pairs that are NOT in the training set
testset = trainset.build_anti_testset()
predictions = algo_KNNBaseline.test(testset)

### Top 10 Book Predictions for all the Users

top_n = get_top_n(predictions, n = 10)

top_n

# print recommended items for each user
recommendations = {}
for uid, user_ratings in top_n.items():
    recommendations.update({uid: [iid for (iid, _) in user_ratings]})
predictions = recommendations[269566]#Random User

type(top_n)

Book_Data.columns
books = Book_Data.copy()

books = books[['ISBN', 'Book_Title', 'Book_Author', 'Year_Of_Publication', 'Publisher',
       'Image_URL_S', 'Image_URL_M', 'Image_L']]
books.head()

Book_Ratings_Data.head()

### Predicting Book Ratings for a given User 

ratings2 = pd.DataFrame(Book_Ratings_Data[Book_Ratings_Data['User_ID'] == 277427])
print(ratings2)

ratings2 = pd.merge(ratings2, books, how = 'left', on = 'ISBN')
ratings2

user2 = pd.DataFrame(top_n[277427])
user2.columns = ['ISBN',"PRED_RATING"]
user2.head()

user2 = pd.merge(user2, books, how = 'left', on = 'ISBN')
user2

Here in this block of code, Initially we have predicted a list of 10 Books for each User, and then we have considered one single user and predicted the ratings that he would rate his book suggestions. 
We see that for a User = 277427, Above are his top 10 book recommendations and his predicted ratings for those book recommendations. 

