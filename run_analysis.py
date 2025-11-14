import os
import re
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_colwidth", 120)
RANDOM_STATE = 42


def main():
    # ===== 2. Read Data =====
    DATA_PATH = "data/Tweets.csv"

    if DATA_PATH is None:
        raise FileNotFoundError(
            "Không tìm thấy file Tweets.csv. Hãy kiểm tra lại đường dẫn.\n"
        )

    df = pd.read_csv(DATA_PATH)
    print("Đã đọc dữ liệu:", DATA_PATH)
    print("thông tin dữ liệu")
    df.info()

    # ===== 3. EDA cơ bản =====
    plt.figure(figsize=(6, 4))
    sns.countplot(x="airline_sentiment", data=df, palette="Set2")
    plt.title("Phân bố cảm xúc người dùng")
    plt.xlabel("Sentiment")
    plt.ylabel("Số lượng")
    plt.show()

    print("Số lượng theo sentiment:\n", df["airline_sentiment"].value_counts())

    # ===== 4. Làm sạch text =====
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", " ", text)       # bỏ URL
        text = re.sub(r"@\w+", " ", text)          # bỏ mention
        text = re.sub(r"#", " ", text)             # bỏ dấu #
        text = re.sub(r"[^a-z\s]", " ", text)      # bỏ ký tự không phải chữ
        text = re.sub(r"\s+", " ", text).strip()   # bỏ khoảng trắng dư
        return text

    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    print("\nVí dụ text sau khi clean:")
    print(df[["text", "clean_text"]].head())

    # ===== 5. WordCloud cho từng sentiment =====
    for sentiment in df["airline_sentiment"].unique():
        text_blob = " ".join(df.loc[df["airline_sentiment"] == sentiment, "clean_text"])
        if len(text_blob) < 10:  # tránh lỗi với nhóm quá ít
            continue
        plt.figure(figsize=(7, 4))
        wc = WordCloud(width=900, height=400, background_color="white").generate(text_blob)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud – {sentiment}")
        plt.show()

    # ===== 6. Phân tích mở rộng =====
    # 6.1 Sentiment theo airline
    plt.figure(figsize=(10, 6))
    sentiment_by_airline = df.groupby(["airline", "airline_sentiment"]).size().unstack(fill_value=0)
    sentiment_by_airline.plot(
        kind="bar", stacked=True, figsize=(10, 6), colormap="coolwarm"
    )
    plt.title("Tỷ lệ sentiment theo hãng hàng không")
    plt.ylabel("Số lượng tweet")
    plt.xlabel("Hãng hàng không")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 6.2 Sentiment theo giờ
    if "tweet_created" in df.columns:
        df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce")
        df["hour"] = df["tweet_created"].dt.hour
        plt.figure(figsize=(12, 5))
        sns.countplot(
            data=df,
            x="hour",
            hue="airline_sentiment",
            palette="Set2",
            order=sorted(df["hour"].dropna().unique()),
        )
        plt.title("Cảm xúc theo giờ đăng tweet")
        plt.xlabel("Giờ (0–23)")
        plt.ylabel("Số lượng tweet")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        plt.show()

    # 6.3 Top múi giờ
    if "user_timezone" in df.columns:
        top_tz = df["user_timezone"].value_counts().head(10)
        if len(top_tz) > 0:
            plt.figure(figsize=(9, 4))
            sns.barplot(x=top_tz.values, y=top_tz.index, palette="viridis")
            plt.title("Top 10 múi giờ của người dùng")
            plt.xlabel("Số lượng")
            plt.ylabel("Múi giờ")
            plt.tight_layout()
            plt.show()

    # 6.4 Lý do tiêu cực
    if "negativereason" in df.columns:
        neg_reason = df["negativereason"].value_counts().head(12)
        if len(neg_reason) > 0:
            plt.figure(figsize=(9, 5))
            sns.barplot(y=neg_reason.index, x=neg_reason.values, palette="Reds_r")
            plt.title("Top lý do người dùng phàn nàn (negative)")
            plt.xlabel("Số lượng tweet")
            plt.ylabel("Lý do")
            plt.tight_layout()
            plt.show()

    # 6.5 Độ dài tweet theo sentiment
    df["text_length"] = df["clean_text"].apply(lambda s: len(s.split()))
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="airline_sentiment", y="text_length", data=df, palette="Set2")
    plt.title("Phân bố độ dài tweet theo cảm xúc")
    plt.xlabel("Sentiment")
    plt.ylabel("Số từ")
    plt.tight_layout()
    plt.show()

    # ===== 7. TF-IDF + Train/Test Split =====
    X_text = df["clean_text"]
    y = df["airline_sentiment"]

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ===== 8. Train & So sánh nhiều model =====
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "Linear SVM": LinearSVC(),
    }

    results = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = round(accuracy_score(y_test, pred), 4)
        results.append((name, acc))
        print(f"\n================= {name} =================")
        print("Accuracy:", acc)
        print(classification_report(y_test, pred))

    res_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(
        "Accuracy", ascending=False
    )
    print("\nSo sánh Accuracy:")
    print(res_df)

    # ===== 9. Confusion Matrix model tốt nhất =====
    best_model_name = res_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_,
    )
    plt.title(f"Confusion Matrix – {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ===== 10. PCA 2D demo =====
    SAMPLE_FOR_PCA = min(3000, X_tfidf.shape[0])
    idx_sample = np.random.RandomState(RANDOM_STATE).choice(
        X_tfidf.shape[0], SAMPLE_FOR_PCA, replace=False
    )
    X_sample = X_tfidf[idx_sample].toarray()
    y_sample = y.iloc[idx_sample].values

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], hue=y_sample, alpha=0.5, s=25, palette="Set2"
    )
    plt.title("Biểu diễn PCA 2D của tweet theo sentiment")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ===== 11. Dự đoán thử tweet mới =====
    sample_tweets = [
        "The flight was delayed for 3 hours, terrible experience.",
        "Amazing service! The crew was so kind and helpful.",
        "It was okay, nothing special.",
        "Customer service solved my issue quickly, thanks!",
    ]

    tfidf_full = TfidfVectorizer(max_features=5000, stop_words="english")
    X_full = tfidf_full.fit_transform(X_text)
    best_model.fit(X_full, y)

    sample_vec = tfidf_full.transform(sample_tweets)
    sample_pred = best_model.predict(sample_vec)
    print("\nDự đoán ví dụ:")
    for t, p in zip(sample_tweets, sample_pred):
        print(f"- {t}  →  {p}")

    # ===== 12. Summary để copy vào báo cáo =====
    summary = {
        "Best Model": best_model_name,
        "Best Accuracy": float(res_df.iloc[0]["Accuracy"]),
        "Negative Ratio": float((df["airline_sentiment"] == "negative").mean()),
        "Mean Tweet Length": float(df["text_length"].mean()),
    }
    print("\n===== SUMMARY (copy vào báo cáo) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
