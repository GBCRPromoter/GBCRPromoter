
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression, SGDRegressor, HuberRegressor, PassiveAggressiveRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
import pickle
import joblib
from sklearn.utils import shuffle
import time
from datetime import datetime
import random
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import make_scorer
from FeatureEngineering_success import extract_features

# 设置全局随机种子

# 加载数据
fileName = 'TD'     # promoter_change   TD
data = pd.read_pickle(f'../data/{fileName}.pkl')


X = data['promoter']
y = data['strength']

X = pd.read_pickle(f'../FeatureEngineering/result_feature/{fileName}_all_feature_X_df.pkl')

# 对 X 进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# # 对 y 进行对数变换
# # y = np.array(y).astype(np.float32)
# y_train_transformed = np.log2(y + 1e-5)
y = np.log2(y + 1e-5)


k_best_percent = 0.2  # 1 0.2
# # 方法1：使用多种特种选择方法选择重要特征
# selector = selection_methods['f_classif']
# X_selected = selector.fit_transform(X, y)

# 方法2：使用SelectKBest选择重要特征
k_best = int(k_best_percent * X.shape[1])  # 0.2   100  # 选择前100个特征
X_selected = SelectKBest(f_regression, k=k_best).fit_transform(X, y)

# 方法3：使用SelectFromModel基于随机森林的特征重要性进行选择
# selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="mean")
# selector.fit(X, y)
# X_selected = selector.transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)

# 定义回归模型和超参数网格
models_and_params = {
    'RandomForestRegressor_singleSearch': (RandomForestRegressor(), {}),
}

# 保存符合条件的结果
valid_results = []
threshold_r2 = 0.5

results = {}

# 定义自定义评分函数，计算 Pearson 相关系数
def pearson_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]  # pearsonr 返回的是相关系数和 p-value, 我们只需要相关系数


for name, (model, params) in models_and_params.items():
    try:
        print(f'Fitting {name}')
        st = time.time()
        # 使用 make_scorer 来创建一个评分对象
        pearson_scorer_func = make_scorer(pearson_scorer)
        # # 使用 GridSearchCV 进行超参数搜索
        grid_search = GridSearchCV(model, params, cv=2, scoring=pearson_scorer_func, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = grid_search.best_estimator_

        # 预测
        y_pred = best_model.predict(X_test)
        # y_pred_all = best_model.predict(X)
        y_pred_train = best_model.predict(X_train)
        y_pred_all = best_model.predict(X_selected)

        # 计算回归误差和R^2
        spearman_corr, p_value = spearmanr(y_test, y_pred)
        spearman_corr_train, p_value_train = spearmanr(y_train, y_pred_train)
        spearman_corr_all, p_value_all = spearmanr(y, y_pred_all)
        pearsonr_corr, pearsonr_p_value = pearsonr(y_test, y_pred)
        pearsonr_corr_train, pearsonr_p_value_train = pearsonr(y_train, y_pred_train)
        pearsonr_corr_all, pearsonr_p_value_all = pearsonr(y, y_pred_all)

        # 合并 y 和 y_pred_all 保持索引一致
        y_true_pred_all = pd.DataFrame({'y_true': y, 'y_pred_all': y_pred_all.ravel()}, index=y.index)
        y_test_pred = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        y_train_pred = pd.DataFrame({'y_test': y_train, 'y_pred': y_pred_train})
        # 存储结果
        results[name] = {
            'spearman_corr': spearman_corr, 'p_value': p_value, 'spearman_corr_train': spearman_corr_train, 'p_value_train': p_value_train,
            'spearman_corr_all': spearman_corr_all, 'p_value_all': p_value_all,
            'pearsonr_corr': pearsonr_corr, 'pearsonr_p_value': pearsonr_p_value, 'pearsonr_corr_train': pearsonr_corr_train, 'pearsonr_p_value_train': pearsonr_p_value_train,
            'pearsonr_corr_all': pearsonr_corr_all, 'pearsonr_p_value_all': pearsonr_p_value_all,
            # 'feature_importances': feature_importances,
        }
    except:
        print(f"{name} 模型训练失败")
    print("-" * 60)

print('done')

