from pyngrok import ngrok
from flask import Flask
from flask_ngrok import run_with_ngrok
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_id = "Bong9/easydata"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()

def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=50,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    return tokenizer.decode(gened[0])
    
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
  # JSON 데이터에서 필요한 정보 추출
  quest = data.get('question')
  full_text = gen(quest) 
  start_index = full_text.find("### 답변:") + len("### 답변:")
  answer_content = full_text[start_index:].strip()
  
  return jsonify(answer_content) 
           
if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = '5000' ,debug=True)
