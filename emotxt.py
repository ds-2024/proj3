from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I feel lonely"]

# 감정 분류 수행
model_outputs = classifier(sentences)
print(model_outputs)  # 리스트 전체 출력
print(model_outputs[0])  # 첫 번째 결과 출력
