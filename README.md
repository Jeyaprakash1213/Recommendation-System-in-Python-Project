🎬 Recommendation System using Python

> Last Updated: 07 Jul, 2025
> A user-personalized recommendation engine using Collaborative Filtering and KNN.
---
📌 About the Project

In this project, I’ve developed a movie recommendation system using Python that recommends similar movies to users based on their rating history. The engine is built using Collaborative Filtering with K-Nearest Neighbors (KNN) and can be used in real-world applications like Netflix or Amazon.

---

🧠 What Is a Recommendation System?
Recommendation systems help users discover items that match their preferences using algorithms. This project focuses on **Collaborative Filtering**, where recommendations are made based on user-item interactions (ratings), rather than the content of the items.

---
🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy – Data wrangling
* Scikit-learn – KNN algorithm
* Matplotlib & Seaborn – Data visualization
* SciPy – Sparse matrix
---
📂 Dataset Description
We used two datasets:
* `ratings.csv`: Contains `userId`, `movieId`, and `rating`
* `movies.csv`: Contains `movieId`, `title`, and `genres`

Loading the data:
```python
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
```
---

🔍 Data Analysis & Preprocessing

1. Basic Statistics
We analyze the number of users, movies, and average rating frequency.

```python
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
```
📸 Screenshot
![Stats](Screenshots/3.png)

---
 2. User Rating Frequency
Shows how many movies each user rated:

```python
user_freq = ratings.groupby('userId')['movieId'].count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())
```
---
3. Best & Worst Rated Movies

Get movies with the highest and lowest average ratings:

```python
mean_rating = ratings.groupby('movieId')['rating'].mean()
highest_rated = mean_rating.idxmax()
lowest_rated = mean_rating.idxmin()

movies.loc[movies['movieId'] == highest_rated]
```
📸 Screenshot
![Movie Ratings](Screenshots/4.png)

---
🧱 Building the User-Item Matrix

We create a sparse matrix of movie-user ratings using SciPy:
```python
from scipy.sparse import csr_matrix

def create_matrix(df):
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper

X, user_mapper, movie_mapper = create_matrix(ratings)
```
---
 🤖 Finding Similar Movies

Using K-Nearest Neighbors and cosine similarity, we find similar movies:
```python
from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k=10):
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    kNN = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    kNN.fit(X)
    
    movie_vec = movie_vec.reshape(1, -1)
    neighbours = kNN.kneighbors(movie_vec, return_distance=False).flatten()
    return [movie_inv_mapper[i] for i in neighbours if movie_inv_mapper[i] != movie_id]
```
---
 🎯 Personalized Recommendations

Recommending movies based on a user’s highest-rated movie:
```python
def recommend_movies_for_user(user_id, X, k=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    fav_movie_id = user_ratings.loc[user_ratings['rating'].idxmax()]['movieId']
    
    similar_ids = find_similar_movies(fav_movie_id, X, k)
    print(f"Since you liked '{movies.loc[movies['movieId'] == fav_movie_id]['title'].values[0]}', you may also like:")

    for movie_id in similar_ids:
        print(movies.loc[movies['movieId'] == movie_id]['title'].values[0])
```
 Example:

```python
recommend_movies_for_user(user_id=150, X=X, k=10)
```

📸 Output
![Recommendations](Screenshots/5.png)

---
 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your `ratings.csv` and `movies.csv` to the project directory.

4. Run:

   ```bash
   python recommendation_system.py
   ```

---
✅ Features

* Recommend top-k movies based on user history
* Uses KNN and cosine similarity
* Lightweight and scalable
* Customizable for different datasets
---
---
📄 License

This project is licensed under the MIT License.
---
📬 Contact
Have feedback or want to collaborate? Reach out via [LinkedIn](https://linkedin.com/in/cjeyaprakash) or open an issue.

---
