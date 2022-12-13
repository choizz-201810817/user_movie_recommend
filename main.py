#%%
# 순서 : 최종df > 유사도df > 최종df * 유사도df(내적) > 새로운df

from numpy import dot
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


moviePath = 'data\grouplens_movies.csv'
ratingPath = 'data\grouples_ratings.csv'
movies = pd.read_csv(moviePath)
ratings = pd.read_csv(ratingPath)


# %%
print(f"movies shape: {movies.shape}")
print(f"ratinigs shape: {ratings.shape}")

# for i in movies.columns:
#     if i in ratings.columns:
#         print(i)

comCol = [col for col in movies.columns if col in ratings.columns]
# print(comCol)

merge_df = pd.merge(ratings, movies , on=comCol[0], how='left')
print('merged dataframe shape :', merge_df.shape)

pivot_df = pd.pivot_table(merge_df, values='rating', index='userId', columns='title')
print('pivot dataframe shape :', pivot_df.shape, '> 최종 dataframe')
pivot_df.head()

pivot_df = pivot_df.fillna(0)
pivot_df.head()

transpose_df = pivot_df.T
print('transpose dataframe shape :', transpose_df.shape)
transpose_df.head()

cos_sim = cosine_similarity(transpose_df, transpose_df) # row끼리의 유사도 산출

cos_df = pd.DataFrame(data=cos_sim, index=transpose_df.index, columns=transpose_df.index)
print('cosine similarity dataframe shape :', cos_df.shape, '> 유사도 dataframe')
cos_df

# 인셉션과 가장 유사한 영화 top 10
print('\n    -- 인셉션과 가장 유사한 영화 top 10 --')
print(cos_df['Inception (2010)'].sort_values(ascending=False)[1:11])

dot_df = pd.DataFrame(dot(pivot_df, cos_df), index=pivot_df.index, columns=pivot_df.columns)
print('dot dataframe shape :', dot_df.shape)

re_df = dot_df.div(cos_df.abs().sum(axis=1), axis=1)
print('repre dataframe shape :', re_df.shape)
## 최종df, 예측평점 데이터프레임의 mse 계산

# pivot_df의 0이 아닌 위치에서의 mse 구하기
nonzero_index = np.nonzero(pivot_df.values)
mse = mean_squared_error(re_df.values[nonzero_index], pivot_df.values[nonzero_index])
print('pivot_df and repo_df mse :', mse)

# %%
re_df2 = np.zeros(pivot_df.shape)
print('re_df2 shape :', re_df2.shape)

for col in range(pivot_df.shape[1]):
    top20_idx = np.argsort(cos_df.values[:,col])[:-21:-1]
    for row in range(pivot_df.shape[0]):
        re_df2[row, col] = cos_df.values[col, :][top20_idx].dot(pivot_df.values[row,:][top20_idx])
        re_df2[row, col] /= np.abs(cos_df.values[col,:][top20_idx]).sum()

re_df2 = pd.DataFrame(re_df2, index=pivot_df.index, columns=pivot_df.columns)
print(re_df.shape)

# %%
nonzero_index2 = np.nonzero(pivot_df.values)
mse2 = mean_squared_error(re_df2.values[nonzero_index], pivot_df.values[nonzero_index])
print('pivot_df and repo_df mse :', mse2)

# %%
# 9번 사용자가 어떤 영화를 좋아하는지 확인
user9 = pivot_df.loc[9,:]
print(user9.sort_values(ascending=False)[:10])

id_series = pivot_df.loc[9,:]
already_seen = id_series[id_series > 0].index.tolist()
all_movies = list(pivot_df)
unseen = [i for i in all_movies if i not in already_seen]

# 9번 사용자가 안 본 영화 중에서 10개의 영화를 추천
reco_9user = re_df2.loc[9,unseen].sort_values(ascending=False)[:10]

print('-- 9번 유저가 안 본 추천 영화 10선 --\n')
reco_9user_df = pd.DataFrame(reco_9user)
reco_9user_df