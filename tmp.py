from flask import Flask, request, jsonify

#필요한 모듈 호출
import pandas as pd
app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
            # JSON 데이터를 파싱
            data = request.get_json()
            # JSON 데이터에서 필요한 정보 추출
            user_preferred_places = data.get('userPreferredPlaces')
            
            
            if additional_condition == None:
                return jsonify(recommendations(wantplaces))

            
            
           
if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = '5000' ,debug=True)
