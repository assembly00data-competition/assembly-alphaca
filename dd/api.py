from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
df = pd.read_csv('lawall.csv', encoding='utf-8')
from openai import OpenAI


# 법률안정보에 대해서 물었을 때 가장 유사한 정보 내뿜기
def find_most_similar_row(input_data, df):
    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['17'])

    # 입력 데이터 벡터화
    input_vector = tfidf_vectorizer.transform([input_data])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # 가장 유사한 행의 인덱스 찾기
    most_similar_index = cosine_similarities.argmax()

    # 가장 유사한 행의 정보 추출
    most_similar_row = df.iloc[most_similar_index]

    return f'해당 법률안은 {most_similar_row[2]}인데, {most_similar_row[4]}과 {most_similar_row[7]}가 소관하고 있으며,  상세내용은 {most_similar_row[18]}입니다.'

from flask import Flask, request, jsonify

#필요한 모듈 호출
import pandas as pd
app = Flask(__name__)
@app.get("/")
def read_root():
    return {"Hello": "World"}
    

@app.route('/', methods=['POST'])
def handle_request():
  # JSON 데이터를 파싱
  data = request.get_json()
  df = pd.read_csv('lawall.csv',encoding = 'utf-8')
  # JSON 데이터에서 필요한 정보 추출
  quest = data.get('question')
  iff = data.get('effect')
  

  # 발급받은 API 키 설정
  OPENAI_API_KEY = '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'

# openai API 키 인증
  model = "gpt-3.5-turbo"
# 질문 작성하기
# 메시지 설정하기

  if iff != None:
    query = iff[50:400]
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_API_KEY,
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system","content": "이 법률안의 기대효과에 대해 알려줘"}, {"role": "user","content": query}])
    return jsonify(response.choices[0].message.content.strip())
  else:
        return jsonify(find_most_similar_row(quest, df))
           
if __name__ == '__main__':
 app.run(host = '0.0.0.0' , port = '5000' ,debug=True)

