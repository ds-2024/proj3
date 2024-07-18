from rake_nltk import Rake
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')



# 샘플 데이터
text = "I love programming in Python. Python is a great programming language. I love natural language processing."

# RAKE 초기화
rake_nltk_var = Rake()

# 텍스트에서 키워드 추출
rake_nltk_var.extract_keywords_from_text(text)

# 키워드와 점수 출력
keyword_scores = rake_nltk_var.get_ranked_phrases_with_scores()
for score, keyword in keyword_scores:
    print(f"{keyword}: {score}")
