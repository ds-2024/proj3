from transformers import pipeline

# STEP 2: Pipeline 초기화
classifier = pipeline("text-classification", model="Doowon96/bert-base-finetuned-ynat")

# STEP 3: 뉴스 제목 입력
title = "SSG닷컴, 창사 이래 첫 희망퇴직…이커머스 '칼바람'"

summary = "근속 2년 이상 본사 직원대상, 최대 24개월 특별퇴직금 지급"

body = """
신세계그룹 계열 전자상거래(이커머스) 업체인 SSG닷컴이 법인 설립 이후 처음으로 희망퇴직을 실시한다. \n앞서 이마트와 롯데온 등 희망퇴직을 단행한 바 있다.\n 
소비 심리가 좀처럼 회복되지 않고, 중국 저가 이커머스 업체가 시장 점유율을 높이면서 실적이 저조해진 유통가가 잇따라 몸집 줄이기에 나선 것이다.\n 
5일 유통업계에 따르면 SSG닷컴은 이날 오전 회사 게시판에 희망퇴직을 공지했다. \n
2022년 7월1일 이전 입사한 근속 2년 이상 본사 직원이 대상이다."""


# STEP 4: 분류 수행
result = classifier(title)

# STEP 5: 결과 출력
print(result)
