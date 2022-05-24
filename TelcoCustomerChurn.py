############################
# Telco Churn Prediction
############################

# İş Problemi: şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi beklenmektedir.

# Veri Seti Hikayesi:
# Veri seti 21 değişken 7043 Gözlem değerine sahiptir.

import warnings
import numpy as np
import seaborn as sns
import pandas as pd
from helpers.functions import *
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###################################
# Görev 1: Keşifçi Veri Analizi
###################################

df = pd.read_csv(r"week7-8-9/Hw8/TelcoCustomerChurn/Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Adım 1
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# errors= "coerce" ile veride bulunan boşluk değerlerini nan'a çevirdik.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3
for col in cat_cols:
    cat_summary(df, col)

num_summary(df, num_cols)

# Adım 4
for col in cat_cols:
    print(pd.DataFrame({"TARGET_COUNT": df.groupby(col)["Churn"].count(),
                       "TARGET_RATIO": df.groupby(col)["Churn"].count() / df.shape[0]}), end="\n\n")

# Adım 5
check_outlier(df, num_cols, 0.1, 0.9)
""" Aykırı değer bulunmamaktadır. """

# Adım 6
df.isnull().sum()
""" TotalCharges'da 11 adet eksik veri bulunmaktadır."""


###################################
# Görev 2: Feature Engineering
###################################

# Adım 1
df.dropna(inplace=True)

# LOF
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:20]  # En kötü 10 değer

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[14]

# Aykırı değerlerin df'ten çıkarılması
df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)
"""
14 satır veri LOF'a göre aykırı değer oluşturuyordu. Bu satırlar dataframe'den silindi.
"""

# Adım 2

df.loc[(df["PhoneService"] == "Yes") &
       (df["InternetService"] != "No") &
       (df["OnlineSecurity"] == "Yes") &
       (df["DeviceProtection"] == "Yes") &
       (df["TechSupport"] == "Yes") &
       (df["StreamingTV"] == "Yes") &
       (df["StreamingMovies"] == "Yes"), ["NEW_ALL_SERVICES"]] = 1
df.NEW_ALL_SERVICES.fillna(0, inplace=True)

df.loc[(df["Contract"] == "Month-to-month"), ["NEW_CONTRACT"]] = "Month"
df.NEW_CONTRACT.fillna("Year", inplace=True)

df["NEW_TENURE"] = pd.cut(df["tenure"], bins=[0, 12, 24, 36, 60, 120], labels=["1", "2", "3", "4", "5"])

# Adım 3
cat_cols = cat_cols + ["NEW_CONTRACT"]
cat_cols.remove("Churn")

df_new = one_hot_encoder(df, cat_cols)
df_new["Churn"] = LabelEncoder().fit_transform(df_new["Churn"])

# Adım 4
num_cols = num_cols + ["NEW_TENURE"]
df_new[num_cols] = RobustScaler().fit_transform(df_new[num_cols])


###################################
# Görev 3: Modelleme
###################################

X = df_new.drop("Churn", axis=1)
y = df_new["Churn"]

# Adım 1

knn = KNeighborsClassifier()
lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
ada = AdaBoostClassifier()
svm = SVC()
dt = DecisionTreeClassifier()
gnb = GaussianNB()
lgbm = LGBMClassifier()
xgb = XGBClassifier()
catb = CatBoostClassifier()

model = [knn, lr, rf, gb, ada, svm, dt, gnb, lgbm, xgb, catb]
model_names = ["KNN", "LR", "RF", "GB", "ADA", "SVM", "DT", "GNB", "LGBM", "XGB", "CATB"]


accuracy = []
f1 = []
for i in model:
    result = cross_validate(i, X, y, cv=5, n_jobs=-1, scoring=["accuracy", "f1"], verbose=2)
    accuracy.append(result["test_accuracy"].mean())
    f1.append(result["test_f1"].mean())

acc_df = pd.DataFrame({"Model": model_names, "Accuracy": accuracy, "F1": f1})
acc_df.sort_values(by="Accuracy", ascending=False, inplace=True)
acc_df.reset_index(drop=True, inplace=True)
selected_model = acc_df["Model"][:4]

# Adım 2

lr.get_params()

lr_params = {"max_iter": [50, 100, 250, 1000],
             "penalty": ["l1", "l2", "elasticnet", "none"],
             "tol": [0.00001, 0.0001, 0.001]}

lr_best_grid = GridSearchCV(lr, lr_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

lr_best_grid.best_params_
# {'max_iter': 100, 'penalty': 'none', 'tol': 1e-05}

###################################

gb.get_params()

gb_params = {"loss": ["deviance", "exponential"],
             "learning_rate": [0.01, 0.1],
             "n_estimators": [100, 250, 1000],
             "min_samples_split": [5, 10, 20],
             "max_depth": [3, 5]}

gb_best_grid = GridSearchCV(gb, gb_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

gb_best_grid.best_params_
# {'learning_rate': 0.01,
#  'loss': 'exponential',
#  'max_depth': 3,
#  'min_samples_split': 5,
#  'n_estimators': 1000}

###################################

ada.get_params()

ada_params = {"n_estimators": [50, 100, 250, 1000],
              "learning_rate": [0.01, 0.1, 0.5],
              "algorithm": ["SAMME", "SAMME.R"]}

ada_best_grid = GridSearchCV(ada, ada_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

ada_best_grid.best_params_
# {'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 250}

###################################

svm.get_params()

svm_params = {"tol": [0.00001, 0.0001, 0.001],
              "degree": [2, 3, 4],
              "kernel": ["linear", "poly", "rbf", "sigmoid"]}

svm_best_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)

svm_best_grid.best_params_
# {'degree': 3, 'kernel': 'poly', 'tol': 1e-05}

###################################
# Final Models

lr_final = lr.set_params(**{'max_iter': 100, 'penalty': 'none', 'tol': 1e-05}).fit(X, y)
gb_final = gb.set_params(**{'learning_rate': 0.01, 'loss': 'exponential', 'max_depth': 3,
                            'min_samples_split': 5, 'n_estimators': 1000}).fit(X, y)
ada_final = ada.set_params(**{'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 250}).fit(X, y)
svm_final = svm.set_params(**{'degree': 3, 'kernel': 'poly', 'tol': 1e-05}).fit(X, y)

lr_result = cross_validate(lr_final, X, y, cv=5, n_jobs=-1,
                           scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
lr_result['test_accuracy'].mean()  # 0.8056
lr_result['test_f1'].mean()  # 0.6020
lr_result['test_roc_auc'].mean()  # 0.8457
lr_result['test_recall'].mean()  # 0.5548
lr_result['test_precision'].mean()  # 0.6581

gb_result = cross_validate(gb_final, X, y, cv=5, n_jobs=-1,
                           scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
gb_result['test_accuracy'].mean()  # 0.8069
gb_result['test_f1'].mean()  # 0.5939
gb_result['test_roc_auc'].mean()  # 0.8463
gb_result['test_recall'].mean()  # 0.5328
gb_result['test_precision'].mean()  # 0.6710

ada_result = cross_validate(ada_final, X, y, cv=5, n_jobs=-1,
                            scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
ada_result['test_accuracy'].mean()  # 0.8066
ada_result['test_f1'].mean()  # 0.5946
ada_result['test_roc_auc'].mean()  # 0.8450
ada_result['test_recall'].mean()  # 0.5349
ada_result['test_precision'].mean()  # 0.6692

svm_result = cross_validate(svm_final, X, y, cv=5, n_jobs=-1,
                            scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
svm_result['test_accuracy'].mean()  # 0.8032
svm_result['test_f1'].mean()  # 0.5661
svm_result['test_roc_auc'].mean()  # 0.8059
svm_result['test_recall'].mean()  # 0.4844
svm_result['test_precision'].mean()  # 0.6816


# Adım 3

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(gb_final, X)

selected_cols = X.columns[gb_final.feature_importances_ > 0.05]

###################################

X_new = X[selected_cols]

gb_new_final = gb.set_params(**{'learning_rate': 0.01, 'loss': 'exponential', 'max_depth': 3,
                                'min_samples_split': 10, 'n_estimators': 1000}).fit(X_new, y)

gb_new_result = cross_validate(gb_new_final, X, y, cv=5, n_jobs=-1,
                               scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
gb_new_result['test_accuracy'].mean()  # 0.8071
gb_new_result['test_f1'].mean()  # 0.5943
gb_new_result['test_roc_auc'].mean()  # 0.8466
gb_new_result['test_recall'].mean()  # 0.5333
gb_new_result['test_precision'].mean()  # 0.6712


# BONUS

# Random Oversampling
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_randomover, y_randomover = oversample.fit_resample(X, y)

gb_oversample = gb.set_params(**{'learning_rate': 0.01, 'loss': 'exponential', 'max_depth': 3,
                                 'min_samples_split': 10, 'n_estimators': 1000}).fit(X_randomover, y_randomover)

gb_oversample_result = cross_validate(gb_oversample, X_randomover, y_randomover, cv=5, n_jobs=-1,
                                      scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
gb_oversample_result['test_accuracy'].mean()  # 0.7770
gb_oversample_result['test_f1'].mean()  # 0.7864
gb_oversample_result['test_roc_auc'].mean()  # 0.8613
gb_oversample_result['test_recall'].mean()  # 0.8209
gb_oversample_result['test_precision'].mean()  # 0.7549

###################################
# SMOTE Oversampling
from imblearn.over_sampling import SMOTE
oversample2 = SMOTE()
X_smote, y_smote = oversample2.fit_resample(X, y)

gb_oversample2 = gb.set_params(**{'learning_rate': 0.01, 'loss': 'exponential', 'max_depth': 3,
                                  'min_samples_split': 10, 'n_estimators': 1000}).fit(X_smote, y_smote)

gb_oversample2_result = cross_validate(gb_oversample2, X_smote, y_smote, cv=5, n_jobs=-1,
                                       scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)
gb_oversample2_result['test_accuracy'].mean()  # 0.8085
gb_oversample2_result['test_f1'].mean()  # 0.8137
gb_oversample2_result['test_roc_auc'].mean()  # 0.8959
gb_oversample2_result['test_recall'].mean()  # 0.8461
gb_oversample2_result['test_precision'].mean()  # 0.7865

###################################
# Resampling

df_new["Churn"].value_counts()
df_not_churn = df_new[df_new["Churn"] == 0]
df_not_churn["Churn"].value_counts()
df_not_churn_sample = df_not_churn.sample(1860)
df_churn = df_new[df_new["Churn"] == 1]
df_new2 = pd.concat([df_not_churn_sample, df_churn])
df_new2["Churn"].value_counts()

X_resample = df_new2.drop(["Churn"], axis=1)
y_resample = df_new2["Churn"]

gb_resample = gb.set_params(**{'learning_rate': 0.01, 'loss': 'exponential', 'max_depth': 3,
                               'min_samples_split': 10, 'n_estimators': 1000}).fit(X_resample, y_resample)

gb_resample_result = cross_validate(gb_resample, X_resample, y_resample, cv=5, n_jobs=-1,
                                    scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=2)

gb_resample_result['test_accuracy'].mean()  # 0.7656
gb_resample_result['test_f1'].mean()  # 0.7734
gb_resample_result['test_roc_auc'].mean()  # 0.8448
gb_resample_result['test_recall'].mean()  # 0.8005
gb_resample_result['test_precision'].mean()  # 0.7483

"""
Presicion (Pozitif olarak tahmin edilenlerin gerçek pozitiflere oranı, TP/(TP+FP)) 
Recall (Pozitif olarak tahmin edilmesi gereken değerlerin ne kadarının pozitif tahmin edildiği oranı, TP/(TP+FN))
"""

"""
Dengesiz veri setinin önüne geçmek için yapılan çalışmalarda accuracy değeri düşerken, f1 değerleri büyük oranda 
artmıştır. f1 skorunda gözlenen değişim recall ve precision değerlerinde de görülmüştür.

En iyi durum SMOTE ile gerçekleştirilen yöntemde görülmüştür. Oversampling yöntemleri, resampling yöntemlerine 
kıyasla daha başarılı bir sonuç ortaya koymuştur.
"""








