################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
################################################

# İş Problemi:
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar
# oluşturulacak.

# Veri Seti Hikayesi:
# Veri seti Flo’dan son alışverişlerini 2020 -2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# 13 Değişken 19945 Gözlem içermektedir.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin ilk alışveriş tarihi
# last_order_date: Müşterinin son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformdayaptığıtoplamalışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type: 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı iseA,B şeklinde
#             yazılmıştır.

import numpy as np
import pandas as pd
from helpers.functions import *
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###############################
# Görev 1: Veriyi Hazırlama
###############################

# Adım 1

df_ = pd.read_csv("week7-8-9/Hw9/flo_data_20K.csv")

df = df_.copy()

check_df(df)

# Adım 2

df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["monetary"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Recency, Tenure değişkenleri için sabit gün seçimi yapılıyor.
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
date_max = df["last_order_date"].max()
calculation_date = date_max + pd.DateOffset(days=2)

df["recency"] = (calculation_date - df["last_order_date"]).dt.days
df["tenure"] = (calculation_date - df["first_order_date"]).dt.days/7  # Tenure değeri haftalık oluşturuldu.

unique_categories = [x for x in df["interested_in_categories_12"]]
unique_categories = np.concatenate([x.strip("][").split(", ") if x != "[]" else ["Other"] for x in unique_categories])
unique_categories = list(set(unique_categories))
unique_categories = [x for x in unique_categories if x != "Other"]

for x in unique_categories:
    df[x] = [1 if x in df["interested_in_categories_12"][i] else 0 for i in range(df.shape[0])]


##############################################
# Görev 2: K-Means ile Müşteri Segmentasyonu
##############################################

# Adım 1

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for x in num_cols:
    print(x, check_outlier(df, x, q1=0.01, q3=0.99))

for x in num_cols:
    replace_with_thresholds(df, x, q1=0.01, q3=0.99)


df[num_cols] = RobustScaler().fit_transform(df[num_cols])

# Adım 2

df_new = df.drop(["master_id", "interested_in_categories_12"], axis=1)
df_new = df_new.drop(date_cols, axis=1)
df_new = one_hot_encoder(df_new, cat_cols, drop_first=True)

corr = df_new.corr()
drop_list = high_correlated_cols(df_new, plot=True, corr_th=0.9)
df_new = df_new.drop(drop_list, axis=1)

# Yöntem 1
kmeans = KMeans()
ssd = []  # Sum of Squared Distances
K = range(1, 20)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_new)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Sum of Squared Distances")
plt.title("Elbow Yöntemi")
plt.show()

# Yöntem 2
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_new)
elbow.show()

""" Optimum küme sayısı k = 6 olarak belirlendi. """

# Adım 3

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_new)
clusters_kmeans = kmeans.labels_

df_["kmeans_cluster_no"] = clusters_kmeans

# Adım 4

# Her cluster'da kaç kişi var?
cat_summary(df_, "kmeans_cluster_no")
# Online alışveriş yapanların kanal ve segment başına harcamaları
df_.pivot_table(values="customer_value_total_ever_online", index="kmeans_cluster_no",
                columns="order_channel", aggfunc="mean")
# Offline alışveriş yapanların online kanal ve segment başına harcamaları
df_.pivot_table(values="customer_value_total_ever_offline", index="kmeans_cluster_no",
                columns="order_channel", aggfunc="mean")
# Online alışveriş yapanların kanal ve segment başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_online", index="kmeans_cluster_no",
                columns="order_channel", aggfunc="sum")
# Offline alışveriş yapanların kanal ve segment başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_offline", index="kmeans_cluster_no",
                columns="order_channel", aggfunc="sum")
# Kanal ve store_type başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_offline", index="kmeans_cluster_no",
                columns="store_type", aggfunc="sum")


#############################################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#############################################################

# Adım 1

hc = linkage(df_new, "ward")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı (Method: ward)")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=8)
plt.axhline(y=70, color='r', linestyle='--')   # 10 küme
plt.axhline(y=150, color='g', linestyle='--')  # 5 küme seçildi.
plt.axhline(y=245, color='b', linestyle='--')  # 3 küme
plt.show()

# Adım 2

cluster = AgglomerativeClustering(n_clusters=5, linkage="ward")
clusters = cluster.fit_predict(df_new)
df_["hi_cluster_no"] = clusters

# Adım 3

# Her cluster'da kaç kişi var?
cat_summary(df_, "hi_cluster_no")
# Online alışveriş yapanların kanal ve segment başına harcamaları
df_.pivot_table(values="customer_value_total_ever_online", index="hi_cluster_no",
                columns="order_channel", aggfunc="mean")
# Offline alışveriş yapanların online kanal ve segment başına harcamaları
df_.pivot_table(values="customer_value_total_ever_offline", index="hi_cluster_no",
                columns="order_channel", aggfunc="mean")
# Online alışveriş yapanların kanal ve segment başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_online", index="hi_cluster_no",
                columns="order_channel", aggfunc="sum")
# Offline alışveriş yapanların kanal ve segment başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_offline", index="hi_cluster_no",
                columns="order_channel", aggfunc="sum")
# Kanal ve store_type başına alışveriş sayıları toplamı
df_.pivot_table(values="order_num_total_ever_offline", index="hi_cluster_no",
                columns="store_type", aggfunc="sum")
