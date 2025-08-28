# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("dataset/train.csv")
df.head()

# %%
df.info()

# %%
print(df['event_type'].nunique())
print(df["product_id"].nunique())
print(df["category_id"].nunique())
print(df["user_id"].nunique())
print(df["user_session"].nunique())

# %%
test_df = pd.read_csv("dataset/test.csv")
test_df.head()

# %%
test_df.info()

# %%
print(test_df['event_type'].nunique())
print(test_df["product_id"].nunique())
print(test_df["category_id"].nunique())
print(test_df["user_id"].nunique())
print(test_df["user_session"].nunique())

# %%
df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
test_df["event_time"] = pd.to_datetime(test_df["event_time"], utc=True)

df["hour"] = df["event_time"].dt.hour
df["day"] = df["event_time"].dt.day
df["weekday"] = df["event_time"].dt.weekday  # 0 = Pazartesi, 6 = Pazar
df["is_weekend"] = (df["weekday"] >= 5).astype(int) # Hafta sonu kontrolü

test_df["hour"] = test_df["event_time"].dt.hour
test_df["day"] = test_df["event_time"].dt.day
test_df["weekday"] = test_df["event_time"].dt.weekday  # 0 = Pazartesi, 6 = Pazar
test_df["is_weekend"] = (test_df["weekday"] >= 5).astype(int) # Hafta sonu kontrolü

# %%
# --- 2.1) Ürün ve kategori çeşitliliği + diğer özetler ---
# --- 1) Session bazlı özet ---
agg_features = df.groupby("user_session").agg(
    user_id=("user_id", "first"),          
    n_events=("event_type", "count"),      
    n_products=("product_id", "nunique"),  
    n_categories=("category_id", "nunique"),
    avg_hour=("hour", "mean"),             
    is_weekend=("is_weekend", "max"),      
    session_value=("session_value", "first"),
).reset_index()

# --- 2) Yoğunluk & tekrar oranları ---
agg_features["product_repeat_rate"] = agg_features["n_events"] / (agg_features["n_products"] + 1)
agg_features["category_repeat_rate"] = agg_features["n_events"] / (agg_features["n_categories"] + 1)

# --- 3) Event frekansları ---
event_counts = df.pivot_table(
    index="user_session",
    columns="event_type",
    values="event_time",
    aggfunc="count",
    fill_value=0
).reset_index()

agg_features = agg_features.merge(event_counts, on="user_session", how="left")

# --- 4) Event oranları ---
agg_features["conversion_rate"] = agg_features["BUY"] / (agg_features["n_events"] + 1)
agg_features["add_cart_ratio"] = agg_features["ADD_CART"] / (agg_features["n_events"] + 1)
agg_features["remove_vs_add"] = agg_features["REMOVE_CART"] / (agg_features["ADD_CART"] + 1)
agg_features["view_to_buy_ratio"] = agg_features["BUY"] / (agg_features["VIEW"] + 1)

# --- 5) İlk ve son event tipleri ---
first_last = df.sort_values("event_time").groupby("user_session").agg(
    first_event=("event_type", "first"),
    last_event=("event_type", "last")
).reset_index()

agg_features = agg_features.merge(first_last, on="user_session", how="left")

# --- 6) BUY pozisyonu ---

# 1) Event sırası hesapla
df["event_index"] = df.groupby("user_session").cumcount() + 1  # her session’da event sırası

# 2) Session bazında ilk BUY pozisyonunu bul
buy_pos = df[df["event_type"] == "BUY"].groupby("user_session").agg(
    buy_position=("event_index", "min")
).reset_index()

# 3) Test setine merge et
agg_features = agg_features.merge(buy_pos, on="user_session", how="left")

# 4) Normalleştirilmiş pozisyon
# NaN olanlar = BUY yok, 1.0 ile dolduruyoruz (en sona kadar hiç buy gelmedi)
agg_features["buy_position_norm"] = agg_features["buy_position"] / (agg_features["n_events"] + 1)
agg_features["buy_position_norm"] = agg_features["buy_position_norm"].fillna(1.0)

# 5) Binary flag ekle (BUY var mı yok mu)
agg_features["has_buy"] = agg_features["buy_position"].notna().astype(int)

# %%
# --- 2.2) Ürün ve kategori çeşitliliği + diğer özetler ---
# --- 1) Session bazlı özet ---

agg_features_test = test_df.groupby("user_session").agg(
    user_id=("user_id", "first"),          
    n_events=("event_type", "count"),      
    n_products=("product_id", "nunique"),  
    n_categories=("category_id", "nunique"),
    avg_hour=("hour", "mean"),             
    is_weekend=("is_weekend", "max"),      
).reset_index()

# --- 2) Yoğunluk & tekrar oranları ---
agg_features_test["product_repeat_rate"] = agg_features_test["n_events"] / (agg_features_test["n_products"] + 1)
agg_features_test["category_repeat_rate"] = agg_features_test["n_events"] / (agg_features_test["n_categories"] + 1)

# --- 3) Event frekansları ---
event_counts = df.pivot_table(
    index="user_session",
    columns="event_type",
    values="event_time",
    aggfunc="count",
    fill_value=0
).reset_index()

agg_features_test = agg_features_test.merge(event_counts, on="user_session", how="left")

# --- 4) Event oranları ---
agg_features_test["conversion_rate"] = agg_features_test["BUY"] / (agg_features_test["n_events"] + 1)
agg_features_test["add_cart_ratio"] = agg_features_test["ADD_CART"] / (agg_features_test["n_events"] + 1)
agg_features_test["remove_vs_add"] = agg_features_test["REMOVE_CART"] / (agg_features_test["ADD_CART"] + 1)
agg_features_test["view_to_buy_ratio"] = agg_features_test["BUY"] / (agg_features_test["VIEW"] + 1)

# --- 5) İlk ve son event tipleri ---
first_last = df.sort_values("event_time").groupby("user_session").agg(
    first_event=("event_type", "first"),
    last_event=("event_type", "last")
).reset_index()

agg_features_test = agg_features_test.merge(first_last, on="user_session", how="left")

# --- 6) BUY pozisyonu (düzenlenmiş) ---

# 1) Event sırası hesapla
test_df["event_index"] = test_df.groupby("user_session").cumcount() + 1  # her session’da event sırası

# 2) Session bazında ilk BUY pozisyonunu bul
buy_pos = test_df[test_df["event_type"] == "BUY"].groupby("user_session").agg(
    buy_position=("event_index", "min")
).reset_index()

# 3) Test setine merge et
agg_features_test = agg_features_test.merge(buy_pos, on="user_session", how="left")

# 4) Normalleştirilmiş pozisyon
# NaN olanlar = BUY yok, 1.0 ile dolduruyoruz (en sona kadar hiç buy gelmedi)
agg_features_test["buy_position_norm"] = agg_features_test["buy_position"] / (agg_features_test["n_events"] + 1)
agg_features_test["buy_position_norm"] = agg_features_test["buy_position_norm"].fillna(1.0)

# 5) Binary flag ekle (BUY var mı yok mu)
agg_features_test["has_buy"] = agg_features_test["buy_position"].notna().astype(int)

# %%
# --- ) event_type frekansları (her session içinde kaç kez geçmiş) ---
event_counts = df.pivot_table(
    index="user_session", # her oturum (session) için satır oluşturur.
    columns="event_type", # her farklı event_type (VIEW, ADD_CART, vs.) ayrı sütun olur.
    values="event_time", # sayım yapılacak değer (zaman damgası, yani her satır bir event).
    aggfunc="count",  # her session için event sayısını sayar.
    fill_value=0 # olmayan event tipleri 0 olarak doldurulur.
).reset_index() # tabloyu normal DataFrame formatına döndürür.

event_counts["add_cart_ratio"] = event_counts["ADD_CART"] / (event_counts["VIEW"] + 1)
event_counts["buy_ratio"] = event_counts["BUY"] / (event_counts["VIEW"] + 1)
event_counts["remove_vs_add"] = event_counts["REMOVE_CART"] / (event_counts["ADD_CART"] + 1)

agg_features = agg_features.merge(event_counts, on="user_session", how="left")

event_counts_test = test_df.pivot_table(
    index="user_session", # her oturum (session) için satır oluşturur.
    columns="event_type", # her farklı event_type (VIEW, ADD_CART, vs.) ayrı sütun olur.
    values="event_time", # sayım yapılacak değer (zaman damgası, yani her satır bir event).
    aggfunc="count",  # her session için event sayısını sayar.
    fill_value=0 # olmayan event tipleri 0 olarak doldurulur.
).reset_index() # tabloyu normal DataFrame formatına döndürür.

event_counts_test["add_cart_ratio"] = event_counts_test["ADD_CART"] / (event_counts_test["VIEW"] + 1)
event_counts_test["buy_ratio"] = event_counts_test["BUY"] / (event_counts_test["VIEW"] + 1)
event_counts_test["remove_vs_add"] = event_counts_test["REMOVE_CART"] / (event_counts_test["ADD_CART"] + 1)

agg_features_test = agg_features_test.merge(event_counts_test, on="user_session", how="left")

# %%
# --- 3) Birleştirme ---
train_session = agg_features.merge(event_counts, on="user_session", how="left", sort=False)
test_session = agg_features_test.merge(event_counts_test, on="user_session", how="left", sort=False)

# %%
# --- 4) event_time sütununu datetime'a çevir ---
train_session["hour"] = df["event_time"].dt.hour
train_session["day"] = df["event_time"].dt.day
train_session["weekday"] = df["event_time"].dt.weekday  # 0 = Pazartesi, 6 = Pazar
train_session["is_weekend"] = (df["weekday"] >= 5).astype(int) # Hafta sonu kontrolü

test_session["hour"] = test_df["event_time"].dt.hour
test_session["day"] = test_df["event_time"].dt.day
test_session["weekday"] = test_df["event_time"].dt.weekday  # 0 = Pazartesi, 6 = Pazar
test_session["is_weekend"] = (test_df["event_time"].dt.weekday >= 5).astype(int) # Hafta sonu kontrolü

# %%
train_session.head()

# %%
test_session.head()

# %%
test_session.shape

# %%
# Özellikler (hedefi ve kimlik kolonlarını çıkartıyoruz)
# Tüm sayısal kolonları seç
feature_cols = train_session.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Hedef ve ID kolonlarını çıkar
feature_cols = [c for c in feature_cols if c not in ['session_value', 'user_id']]
feature_cols

# %%
from sklearn.model_selection import train_test_split

# Bütün session ID'lerini al
sessions = train_session['user_session'].unique()

# Session bazında train/val böl
train_sess, val_sess = train_test_split(sessions, test_size=0.1, random_state=42)

# Split'i uygula
train_data = train_session[train_session['user_session'].isin(train_sess)]
val_data   = train_session[train_session['user_session'].isin(val_sess)]

# Özellikler ve hedef
X_train = train_data[feature_cols]
y_train = train_data['session_value']

X_val   = val_data[feature_cols]
y_val   = val_data['session_value']


# %%
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

X = train_session[feature_cols].values
y = train_session['session_value'].values
X_test = test_session[feature_cols].values
groups = train_session['user_session'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# %%
# --- KFold ayarı ---
gkf = GroupKFold(n_splits=5)

fold = 1
val_scores = []

for train_idx, val_idx in gkf.split(X_scaled, y, groups=groups):
    print(f"\n🔹 Fold {fold} başlıyor...")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_ann = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='linear')  
    ])

    model_ann.compile(
        optimizer=Adam(),
        loss='mse',
        metrics=['mse']
    )

    history = model_ann.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
    )

    val_loss, val_mse = model_ann.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold} MSE: {val_mse:.4f}")
    val_scores.append(val_mse)

    fold += 1

# %%
avg_mse = np.mean(val_scores)
print(f"\n✅ Ortalama MSE: {avg_mse:.4f}")

# %%
test_pred_ann = model_ann.predict(X_test_scaled)

submission_df = pd.DataFrame({
    'user_session': test_session['user_session'],
    'session_value': test_pred_ann.flatten()
})

# %%
original_order = test_df[["user_session"]].drop_duplicates()
submission_df = original_order.merge(submission_df, on="user_session", how="left")

submission_df.to_csv("sample_submission.csv", index=False)
print("sample_submission.csv dosyası kaydedildi ✅")


