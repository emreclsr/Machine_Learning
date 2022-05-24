###################################
# Ev Fiyat Tahmin Modeli
###################################

# İş Problemi:
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veri seti kullanılarak, farklı tipteki evlerin
# fiyatlarına ilişkin bir makine öğrenmesi projesi geliştirilmek istenmektedir.

# Veri Seti Hikayesi:
# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde
# bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir
# kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde
# ev fiyatları boş bırakılmış olup, bu değerleri sizin  tahmin etmeniz beklenmektedir.

# Toplam Gözlem: 1460   Sayısal Değişken: 38   Kategorik Değişken: 43

import warnings
import numpy as np
import pandas as pd
from helpers.functions import *
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from lightgbm import LGBMRegressor


warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("week7-8-9/Hw8/HousePrice/train.csv")

###################################
# Keşifçi Veri Analizi
###################################

check_df(df)

"""
NA olan değerler anlam ifade etmektedir. Silinmemelidir.
Quantiles değerlerine göre aykırı olabilecek değerler vardır. Detaylı inceleme yapılmalıdır.
"""
df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)
df.Alley.fillna("None", inplace=True)
df.fillna({"BsmtQual" : "No_Bsmt",
           "BsmtCond" : "No_Bsmt",
           "BsmtExposure" : "No_Bsmt",
           "BsmtFinType1": "No_Bsmt",
           "BsmtFinType2" : "No_Bsmt"}, inplace=True)
df.FireplaceQu.fillna("No_Fireplace", inplace=True)
df.fillna({"GarageType" : "No_Garage",
           "GarageYrBlt" : "No_Garage",
           "GarageFinish" : "No_Garage",
           "GarageQual": "No_Garage",
           "GarageCond" : "No_Garage"}, inplace=True)
df.PoolQC.fillna("No_Pool", inplace=True)
df.Fence.fillna("No_Fence", inplace=True)
df.MiscFeature.fillna("None", inplace=True)

df.drop(['Id'], axis=1, inplace=True)
df.dropna(inplace=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=20, car_th=25)

be_num_cols = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
               "Fireplaces", "GarageYrBlt", "GarageCars", "PoolArea", "MoSold", "YrSold"]

num_cols = num_cols + be_num_cols
[cat_cols.remove(x) for x in be_num_cols if x in cat_cols]

date_cols = ["GarageYrBlt", "MoSold", "YrSold"]
[num_cols.remove(x) for x in date_cols]

for col in cat_cols:
    cat_summary(df, col)
"""
Rare encoder yapmak gerekecektir.
"""

num_summary(df, num_cols)

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


#####################################
# Özellik Mühendisliği
#####################################

# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:10]  # En kötü 10 değer

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[6]

# Aykırı değerlerin df'ten çıkarılması
df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)
"""
6 satır veri LOF'a göre aykırı değer oluşturuyordu. Bu satırlar dataframe'den silindi.
"""

# Rare Encoder

df_new = rare_encoder(df, 0.03)

for col in cat_cols:
    target_summary_with_cat(df_new, "SalePrice", col)


# Label Encoding (Binary Encoding)

df_new2 = one_hot_encoder(df_new, cat_cols, drop_first=True)
df_new2.reset_index(drop=True, inplace=True)


# Feature Interactions (Özellik Etkileşimleri)

df_new2["NEW_Age"] = df_new2["YrSold"] - df_new2["YearBuilt"]
df_new2.loc[(df_new2["2ndFlrSF"] > 0), "NEW_FLOOR"] = 1  # Duplex
df_new2.loc[(df_new2["2ndFlrSF"] == 0), "NEW_FLOOR"] = 0  # Single
df_new2.drop(["YrSold", "YearBuilt", "MoSold", "GarageYrBlt"], axis=1, inplace=True)

#####################################
# Model Kurma
#####################################

X = df_new2.drop("SalePrice", axis=1)
y = df_new2["SalePrice"]


################################################
# Random Forests

# --Default Model--

rf_model = RandomForestRegressor(random_state=55)
cv_results = cross_validate(rf_model, X, y, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=2)
cv_results['test_neg_root_mean_squared_error'].mean()  # -29950.65
cv_results['test_neg_mean_absolute_error'].mean()  # -17841.93
cv_results['test_r2'].mean()  # 0.8552

################################################
# LightGBM

# --Default Model--

lgbm_model = LGBMRegressor(random_state=55)
cv_results = cross_validate(lgbm_model, X, y, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=2)
cv_results['test_neg_root_mean_squared_error'].mean()  # -28592.22
cv_results['test_neg_mean_absolute_error'].mean()  # -16646.47
cv_results['test_r2'].mean()  # 0.8672


################################################
# Random Forest Grid Search

rf_model.get_params()

rf_params = {"max_depth": [8, 11, 14, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [5, 10, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

rf_best_grid.best_params_
"""
{'max_depth': 14,
 'max_features': 'auto',
 'min_samples_split': 5,
 'n_estimators': 500}
 """

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=2)
cv_results['test_neg_root_mean_squared_error'].mean()  # -29676.62
cv_results['test_neg_mean_absolute_error'].mean()  # -17662.41
cv_results['test_r2'].mean()  # 0.8574

################################################
# LightGBM Forest Grid Search

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1, 0.5],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

lgbm_best_grid.best_params_
"""
{'colsample_bytree': 0.3, 
'learning_rate': 0.1, 
'n_estimators': 1000}
"""

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5,
                            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_jobs=-1, verbose=2)
cv_results['test_neg_root_mean_squared_error'].mean()  # -27527.94
cv_results['test_neg_mean_absolute_error'].mean()  # -16638.03
cv_results['test_r2'].mean()  # 0.8780


################################################

test_df = pd.read_csv("week7-8-9/Hw8/HousePrice/test.csv")

def data_prep(df):
    df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)
    df.Alley.fillna("None", inplace=True)
    df.fillna({"BsmtQual": "No_Bsmt",
               "BsmtCond": "No_Bsmt",
               "BsmtExposure": "No_Bsmt",
               "BsmtFinType1": "No_Bsmt",
               "BsmtFinType2": "No_Bsmt"}, inplace=True)
    df.FireplaceQu.fillna("No_Fireplace", inplace=True)
    df.fillna({"GarageType": "No_Garage",
               "GarageYrBlt": "No_Garage",
               "GarageFinish": "No_Garage",
               "GarageQual": "No_Garage",
               "GarageCond": "No_Garage"}, inplace=True)
    df.PoolQC.fillna("No_Pool", inplace=True)
    df.Fence.fillna("No_Fence", inplace=True)
    df.MiscFeature.fillna("None", inplace=True)

    df.drop(['Id'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df_new = rare_encoder(df, 0.03)

    df_new2 = one_hot_encoder(df_new, cat_cols, drop_first=True)
    df_new2.reset_index(drop=True, inplace=True)

    df_new2["NEW_Age"] = df_new2["YrSold"] - df_new2["YearBuilt"]
    df_new2.loc[(df_new2["2ndFlrSF"] > 0), "NEW_FLOOR"] = 1  # Duplex
    df_new2.loc[(df_new2["2ndFlrSF"] == 0), "NEW_FLOOR"] = 0  # Single
    df_new2.drop(["YrSold", "YearBuilt", "MoSold", "GarageYrBlt"], axis=1, inplace=True)

    return df_new2


df_test = data_prep(test_df)

rf_y_test = rf_final.predict(df_test)
lgbm_y_test = lgbm_final.predict(df_test)

"""
one_hot_encoder kullanıldığı için model ve test verisindeki kolon sayıları eşit olmuyor. 
Bu nedenle predict değerleri elde edilemedi.
"""








