import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 샘플 데이터
documents = ["I love programming in Python", "Python is a great programming language", "I love natural language processing"]

# TF-IDF 벡터라이저 초기화
tfidf_vectorizer = TfidfVectorizer()

# 문서에 TF-IDF 적용
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 결과를 데이터프레임으로 변환
import pandas as pd
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(tfidf_df)
