#####################################
# Makine Öğrenmesi ile Maaş Tahmini
#####################################

# İş Problemi:
# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir
# makine öğrenmesi modeli geliştiriniz.

# Veri Seti Hikayesi:
# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır. Veri seti
# 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri orijinal olarak Sports
# Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing
# Company, New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

# Değişkenler:
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptığında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyuncunun kariyeri boyunca yaptığı en değerli sayı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assists: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

#####################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def check_df(dataframe, head=5):
    print(" Shape ".center(40, "#"))
    print(dataframe.shape)
    print(" Types ".center(40, "#"))
    print(dataframe.dtypes)
    print(" Head ".center(40, "#"))
    print(dataframe.head(head))
    print(" Tail ".center(40, "#"))
    print(dataframe.tail(head))
    print(" NA ".center(40, "#"))
    print(dataframe.isnull().sum())
    print(" Quantiles ".center(40, "#"))
    print(dataframe.describe([0, 0.01, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float",
                                                                                              np.dtype('int64'),
                                                                                              np.dtype('float64')]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float", np.dtype('int64'), np.dtype('float64')]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


#####################################
# Keşifçi Veri Analizi
#####################################

df = pd.read_csv("week7/Hw7/Maas_Tahmin_Modeli/hitters.csv")

check_df(df)

"""
shape: (322,20), 
3 object, 1 float, 16 int
salary verisinde 59 nan değer var
quantiles incelemesine göre bazı verilerde aykırılık var gibi duruyor detaylı analliz gerekli
"""

# Hedef Değişken Analizi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Bağımlı değişken Salary num_cols içerisinden çıkarılıyor.
num_cols = [col for col in num_cols if col not in "Salary"]

for col in cat_cols:
    cat_summary(df, col)

num_summary(df, num_cols)

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


#####################################
# Özellik Mühendisliği
#####################################

# Aykırı Değer Analizi

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

"""
Aykırı değer bulunmamaktadır.
"""

# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5]  # En kötü 5 değer

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]

# Aykırı değerlerin df'ten çıkarılması
df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)
"""
3 satır veri LOF'a göre aykırı değer oluşturuyordu. Bu satırlar dataframe'den silindi.
"""

# Eksik Değer Analizi

# Eksik değer oranı
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


cor = df.corr()

df.groupby(["League", "Division"])["Salary"].median()
df.groupby(["League", "Division"])["Salary"].mean()

df["Salary"].fillna(df.groupby(["League", "Division"])["Salary"].transform("median"), inplace=True)

"""
Salary verisindeki eksik verileri doldurmak için öncelikle numerik kolonlar içerisinde korelasyonu yüksek bir veri 
arandı. Fakat yüksek korelasyone sahip bir kolon bulunamadı. Bunun ardından kategorik verilere göre bir değerlendirme 
yapıldı. Salary verisi normal dağılmadığından medyan değerleri ile eksik değerler doldurulacak.
"""


# Label Encoding (Binary Encoding)

"""
Normalde veride 2 sınıf olduğundan LabelEncoder kullanılırdı fakat one_hot_encoder içerisine yer alan get_dummies
metodu drop_first=True ile kullanılırsa benzer işi yapmaktadır.
"""
df_new = one_hot_encoder(df, cat_cols, drop_first=True)
df_new.reset_index(drop=True, inplace=True)


# Feature Interactions (Özellik Etkileşimleri)

df_new["New_LeagueChange"] = [0 if df_new["League_N"][i] == df_new["NewLeague_N"][i] else 1
                              for i in range(df_new.shape[0])]

# df_new["New_LeagueChange"] = [0 if df_new["League_N"][i] == df_new["NewLeague_N"][i] else -1
#                              if df_new["League_N"][i] > df_new["NewLeague_N"][i] else 1
#                              for i in range(df_new.shape[0])]

df_new["New_Walks_RBI"] = df_new["Walks"] / (df_new["RBI"]+1)

for i in ["CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks"]:
    df_new[f"New_{i}"] = df_new[i] / df_new["Years"]

df_new["New_AtBat_Hits"] = df_new["AtBat"] / df_new["Hits"]

df_new["New_HmRun_Runs"] = df_new["HmRun"] / (df_new["Runs"]+1)

#df_new["New_RunsAssists_Errors"] = (df_new["Runs"] + df_new["Assists"]) / (df_new["Errors"]+1)

df_new["New_RunsAssists"] = df_new["Runs"] + df_new["Assists"]

#df_new["New_Score"] = (df_new["RBI"] + df_new["Assists"] + df_new["Walks"] - df_new["Errors"]) / df_new["AtBat"]

df_new.loc[(df_new["Years"] <= 2), "New_Years"] = 0
df_new.loc[(df_new["Years"] > 2) & (df_new['Years'] <= 5), "New_Years"] = 1
df_new.loc[(df_new["Years"] > 5) & (df_new['Years'] <= 10), "New_Years"] = 2
df_new.loc[(df_new["Years"] > 10) & (df_new['Years'] <= 15), "New_Years"] = 3
df_new.loc[(df_new["Years"] > 15), "New_Years"] = 4


# Feature Scaling (Özellik Ölçeklendirme)

sca_cols = [col for col in df_new.columns if col not in ["Salary", "League_N", "Division_W", "NewLeague_N",
                                                         "NewLeagueChange"]]

rs = RobustScaler()
df_new[sca_cols] = rs.fit_transform(df_new[sca_cols])


# Korelasyon analizi
corr_matrix = df_new.corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr_matrix, cmap="RdBu")
plt.show()

# Yüksek korelasyonlu değişkenlerin silinmesi
drop_list = high_correlated_cols(df, corr_th=0.85)
df_new.drop(drop_list, axis=1, inplace=True)


# Linear Regression

X = df_new.drop("Salary", axis=1)
y = df_new["Salary"]

reg_model = LinearRegression().fit(X, y)

reg_model.intercept_
reg_model.coef_


np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=20,
                                 scoring="neg_mean_squared_error")))

np.sqrt(mean_squared_error(y, reg_model.predict(X)))
mean_absolute_error(y, reg_model.predict(X))

"""
#########################################################################
# SGD Regressor

sgdr = SGDRegressor(max_iter=10000)

parameters = {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
              "epsilon": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
              "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
              "penalty": ["l2", "l1", "elasticnet"]}

clf = GridSearchCV(sgdr, parameters, verbose=2)
clf.fit(X, y)

sorted(clf.cv_results_.keys())

clf.best_estimator_

sgdr = SGDRegressor(alpha=0.05, epsilon=0.2, eta0=0.05, max_iter=10000, penalty='elasticnet')
sgdr.fit(X, y)

np.mean(np.sqrt(-cross_val_score(sgdr, X, y, cv= 20, scoring="neg_mean_squared_error")))
np.sqrt(mean_squared_error(y, sgdr.predict(X)))
mean_absolute_error(y, sgdr.predict(X))
"""


def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(20, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


plot_feature_importance(reg_model.coef_, X.columns, "Linear Regression")





