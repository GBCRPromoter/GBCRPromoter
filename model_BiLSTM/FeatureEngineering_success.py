
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from transformers import BertTokenizer, BertModel
# from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec, FastText
from gensim.downloader import load as api_load
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
import torch    # https://pytorch.org/
# ys_pc: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# 自定义评估函数
def evaluate_model(y_test, y_pred, y_prob):
    # 计算各项指标
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    res = {'Recall': recall, 'Precision': precision, 'Accuracy': accuracy, 'F1 Score': f1, 'MCC': mcc,
           'Specificity': specificity, 'AUROC': auroc, 'AUPRC': auprc}
    return res

# 使用五折交叉验证对训练集进行评估
def cross_validation(clf, X_train, y_train):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy')
    print(f'五折交叉验证平均准确率：{np.mean(cv_scores) * 100:.2f}%')

# 特征提取方法和对应的处理流程
def extract_features(method, corpus):
    # 1. CountVectorizer，Grams / n-grams 特征提取
    # 使用 CountVectorizer 对氨基酸序列进行数值化表示，类似于词袋模型（Bag of Words）
    # 每个氨基酸视为一个特征，这里将每个序列按氨基酸频率表示
    if method == "CountVectorizer":
        vectorizer = CountVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus).toarray()

    elif method == "CountVectorizer_n_grams":   # Grams / n-grams 特征提取
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
        X = vectorizer.fit_transform(corpus).toarray()

    elif method == "TfidfVectorizer":   # TF-IDF
        vectorizer = TfidfVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus).toarray()

    elif method == "HashingVectorizer":
        vectorizer = HashingVectorizer(analyzer='char', n_features=1000)
        X = vectorizer.fit_transform(corpus).toarray()

    elif method == "LDA":  # Latent Dirichlet Allocation (LDA)
        vectorizer = CountVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus)
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        X = lda.fit_transform(X)

    elif method == "LSA":  # Latent Semantic Analysis (LSA)
        vectorizer = TfidfVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus)
        svd = TruncatedSVD(n_components=min(X.shape[0], X.shape[1]))
        X = svd.fit_transform(X)

    elif method == "PCA":  # PCA (Principal Component Analysis)
        vectorizer = TfidfVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus).toarray()
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        X = pca.fit_transform(X)

    elif method == "t-SNE":  # t-SNE (t-distributed Stochastic Neighbor Embedding)
        vectorizer = TfidfVectorizer(analyzer='char')
        X = vectorizer.fit_transform(corpus).toarray()
        tsne = TSNE(n_components=2, random_state=42)
        X = tsne.fit_transform(X)

    elif method == "Word2Vec":  # Word2Vec Embeddings
        sentences = [list(seq) for seq in corpus]   # 将每个序列视为一个单词序列
        w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        # X = np.array([np.mean([w2v_model.wv[word] for word in seq if word in w2v_model.wv], axis=0) for seq in sentences])
        X = np.array([np.mean([w2v_model.wv[word] for word in seq if word in w2v_model.wv], axis=0) if seq else np.zeros(100) for seq in sentences])

    elif method == "FastText":  # FastText Embeddings
        sentences = [list(seq) for seq in corpus]
        ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
        # X = np.array([np.mean([ft_model.wv[word] for word in seq if word in ft_model.wv], axis=0) for seq in sentences])
        X = np.array([np.mean([ft_model.wv[word] for word in seq if word in ft_model.wv], axis=0) if seq else np.zeros(100) for seq in sentences])

    elif method == "Doc2Vec":  # Doc2Vec Embeddings
        documents = [TaggedDocument(list(seq), [i]) for i, seq in enumerate(corpus)]
        doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
        X = np.array([doc2vec_model.infer_vector(list(seq)) for seq in corpus])

    elif method == "GloVe":  # GloVe Embeddings
        glove_model = api_load("glove-wiki-gigaword-100")
        # X = np.array([np.mean([glove_model[word] for word in seq if word in glove_model], axis=0) if seq else np.zeros(100) for seq in corpus])
        X = []
        for seq in corpus:
            embeddings = [glove_model[word] for word in seq if word in glove_model]
            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
            else:
                mean_embedding = np.zeros(100)  # 假设 GloVe 的嵌入维度为100
            X.append(mean_embedding)
        X = np.array(X)
    elif method == "BERT":  # BERT Embeddings (使用 Transformers 库)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        X = []
        for seq in corpus:
            inputs = tokenizer(seq, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            X.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        X = np.array(X)

    # elif method == "Sentence-BERT":  # Sentence-BERT Embeddings
    #     sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    #     X = sbert_model.encode(corpus)

    return X

if __name__ == '__main__':
    datasets = pd.read_csv(f"../features_ys/train_src.csv", low_memory=False)     # , header=None
    # datasets.insert(0, 'label', pd.Series(np.concatenate([np.zeros(5112), np.ones(7462)])))
    # datasets.insert(0, 'label', pd.Series(np.concatenate([np.zeros(3721), np.ones(2358)])))
    datasets.insert(0, 'label', pd.Series(np.concatenate([np.zeros(3439), np.ones(2213)])))
    # X_train, y_train = np.array(datasets.iloc[:, 1:]), np.array(datasets.iloc[:, 0])
    X_train, y_train = datasets.iloc[:, 1:], datasets.iloc[:, 0]

    # X_test = pd.read_csv("../0project_ys/features_ys/test.csv", header=None, index_col=0)
    X_test = pd.read_csv("../features_ys/test_src.csv")
    # y_test = pd.Series(np.concatenate([np.zeros(len(X_test) // 2), np.ones(len(X_test) - len(X_test) // 2)]))
    y_test = pd.Series(np.concatenate([np.zeros(382), np.ones(245)]))
    X, y = X_train, y_train

    # 将训练集和测试集的序列组合
    corpus = pd.concat([X_train, X_test], axis=0).iloc[:, 0].tolist()


    # 遍历所有特征提取方法
    methods = ["CountVectorizer", "CountVectorizer_n_grams", "TfidfVectorizer", "HashingVectorizer", "LDA", "LSA", "PCA", "t-SNE", "Word2Vec", "FastText", "Doc2Vec", "GloVe", "BERT", "Sentence-BERT"]

    for method in methods:
        X = extract_features(method, corpus)

        X_train, X_test = X[:len(X_train)], X[len(X_train):]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cross_validation(clf, X_train, y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        evaluation_results = evaluate_model(y_test, y_pred, y_prob)
        print(f"\n{method} 结果：", evaluation_results)