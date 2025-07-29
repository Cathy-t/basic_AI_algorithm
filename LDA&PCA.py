# encoding: utf-8
# @File  : LDA&PCA.py
# @Author: cathy
# @Desc : 
# @Date  :  2025/07/29


from sklearn.decomposition import PCA
import numpy as np

# 示例数据
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# 创建PCA模型，降维至1维
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("主成分方差解释率:", pca.explained_variance_ratio_)
print("降维后数据形状:", X_reduced.shape)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

# 示例数据（含类别标签）
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])  # 类别标签

# 创建LDA模型，降维至1维
lda = LinearDiscriminantAnalysis(n_components=1)
X_reduced = lda.fit_transform(X, y)

print("类别间分离度:", lda.explained_variance_ratio_)
print("降维后数据形状:", X_reduced.shape)