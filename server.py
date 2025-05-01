"""
영화 정보 및 감정 분석을 위한 FastAPI 서버
"""

import sys
import torch
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from kobert_transformers import get_tokenizer
from transformers import BertForSequenceClassification
from chromadb import HttpClient

import torch.nn.functional as F

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

# ===== 모델 및 데이터베이스 설정 =====
# KoBERT 모델 로드
from dotenv import load_dotenv
import os
from openai import OpenAI
import uuid
from langchain_core.runnables import RunnableSequence

# 환경 변수 로드
load_dotenv()

model_path = os.getenv("MODEL_PATH")
kobert_tokenizer = get_tokenizer()
kobert_model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kobert_model.to(device)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = HttpClient(host="chromadb", port=8000)

QUESTION_COLLECTION_NAME = "question-collection"
question_collection = chroma_client.get_or_create_collection(name=QUESTION_COLLECTION_NAME)


movie_collection = chroma_client.get_or_create_collection(name="movie-collection")

# 영화 제목 캐시
movie_titles = set()
for metadata in movie_collection.get()["metadatas"]:
    title = metadata.get("title")
    if title:
        movie_titles.add(title)

# ===== FastAPI 앱 설정 =====
app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://ownarea.synology.me"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 데이터 모델 =====
class MessageType(BaseModel):
    type: Literal["감정", "대화", "질의", "목록"] = Field(description="메시지의 유형")
    confidence: float = Field(description="분류 신뢰도")

class ChatRequest(BaseModel):
    message: str

# ===== 유틸리티 함수 =====
def normalize_text(text: str) -> str:
    """
    텍스트를 정규화하는 함수
    - 괄호 안의 내용 제거
    - 공백 제거
    - 소문자 변환
    """
    text = re.sub(r"\(.*?\)", "", text) # 괄호 안의 내용 제거
    text = re.sub(r"\s+", "", text) # 공백 제거
    return text.lower() # 소문자 변환

def extract_title_from_query(query: str) -> str | None: # 영화 제목 추출(튜플 형식으로 반환)
    """
    쿼리에서 영화 제목을 추출하는 함수
    - 정규화된 쿼리와 키워드를 비교하여 매칭되는 영화 제목 반환
    """
    normalized_query = normalize_text(query) # 정규화된 쿼리 생성
    result = movie_collection.get(include=["metadatas", "documents"]) # 영화 정보 조회
    
    for doc, meta in zip(result["documents"], result["metadatas"]): # 영화 정보와 메타데이터 동시 순회
        raw_keywords = meta.get("keywords", []) # 키워드 조회
        if isinstance(raw_keywords, str): # 키워드가 문자열인 경우
            try:
                keyword_list = json.loads(raw_keywords)
            except json.JSONDecodeError:
                keyword_list = []
        elif isinstance(raw_keywords, list):
            keyword_list = raw_keywords
        else:
            keyword_list = []
            
        keyword_list = [normalize_text(k) for k in keyword_list if isinstance(k, str)]
        if any(k in normalized_query for k in keyword_list):
            return meta.get("title")
    return None

def predict_sentiment(sentence: str) -> str:
    """
    KoBERT를 사용하여 문장의 감정을 분석하는 함수
    - 긍정/부정/중립 분류 결과 반환
    """
    inputs = kobert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    kobert_model.eval()

    with torch.no_grad():
        outputs = kobert_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)  # 확률값 계산
        positive_prob = probs[0][1].item()
        negative_prob = probs[0][0].item()

    print(f"긍정: {positive_prob}, 부정: {negative_prob}, 가중치: {abs(positive_prob - negative_prob)}")
    # 중립 기준값 설정 (예: 확신이 0.6 미만이면 중립 처리)
    if abs(positive_prob - negative_prob) < 0.8:
        return "중립"
    elif positive_prob > negative_prob:
        return "긍정"
    else:
        return "부정"

def classify_message(message: str) -> MessageType:
    """
    LangChain을 사용하여 메시지 유형을 분류하는 함수
    """
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_client.api_key)
    
    template = """
    다음 메시지를 보고 아래 4가지 중 하나로 분류하세요:

    - 감정: 영화에 관련된 평가가 포함이 되어야 하며 화남, 기쁨, 짜증, 우울, 설렘 등 **감정 상태**를 표현한 문장 (예: "이 영화 내용이 짜증나", "주인공 연기가 넘사벽이야", "스토리가 좀 어정쩡하네")
    - 대화: 인사, 일상 대화, 잡담 등 (예: "안녕?", "밥 먹었어?", "잘 지내?")
    - 질의: **특정 영화 제목을 포함하며**, 해당 영화에 대해 질문하는 문장 (예: "기생충 줄거리 알려줘", "쥬만지 평점은?")
    - 목록: **특정 영화 제목 없이**, 영화 추천이나 목록을 요청하는 문장 (예: "최신 영화 뭐 있어?", "요즘 볼만한 영화 추천해줘")

    메시지: {message}

    {format_instructions}
    """
    
    parser = PydanticOutputParser(pydantic_object=MessageType)
    prompt = PromptTemplate(
        template=template,
        input_variables=["message"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm
    result = chain.invoke({"message": message})
    
    return parser.parse(result.content)

# ===== 영화 정보 관련 함수 =====
def list_movie_titles_from_rag() -> str:
    """
    RAG에 등록된 모든 영화 제목 목록을 반환하는 함수
    """
    result = movie_collection.get(include=["metadatas"])
    titles = [meta.get("title", "(제목 없음)") for meta in result["metadatas"] if meta.get("title")]
    
    if not titles:
        return "등록된 영화가 없습니다."
        
    response = "검색된 영화 목록입니다:\n\n"
    for idx, title in enumerate(sorted(titles), 1):
        response += f"{idx}. {title}\n"
    return response.strip()

def get_movie_detail_by_title(title: str) -> str:
    """
    영화 제목으로 상세 정보를 조회하는 함수
    """
    result = movie_collection.get(include=["documents", "metadatas"])
    for doc, meta in zip(result["documents"], result["metadatas"]):
        if meta.get("title") == title:
            return f"영화 제목: {title}\n\n개봉일: {meta.get('release_date')}\n\n장르: {meta.get('genre')}\n\n국가: {meta.get('country')}\n\n런닝타임: {meta.get('running_time')}분\n\n줄거리:{doc}"
    return "등록된 정보가 없습니다."

def get_movie_detail_or_fallback(query: str) -> tuple[str | None, str]:
    """
    영화 정보를 조회하는 함수
    - 키워드 매칭을 시도하고, 실패 시 fallback 방식으로 검색
    """
    # 키워드 매칭 시도
    extracted_title = extract_title_from_query(query)
    if extracted_title:
        return get_movie_detail_by_title(extracted_title), "RAG 키워드 매칭"
    
    # Fallback: 토큰 기반 검색
    tokens = [t.lower() for t in re.findall(r'[ㄱ-ㅣ가-힣a-zA-Z0-9]+', query)]
    result = movie_collection.get(include=["metadatas", "documents"])
    match_scores = []
    
    for meta, doc in zip(result["metadatas"], result["documents"]):
        raw_keywords = meta.get("keywords", [])
        if isinstance(raw_keywords, str):
            try:
                keyword_list = json.loads(raw_keywords)
            except:
                keyword_list = []
        elif isinstance(raw_keywords, list):
            keyword_list = raw_keywords
        else:
            keyword_list = []
            
        keyword_list = [k.strip().lower() for k in keyword_list]
        score = sum(1 for token in tokens if token in keyword_list)
        if score > 0:
            match_scores.append((score, meta.get("title"), doc))
    
    if match_scores:
        match_scores.sort(reverse=True)
        _, best_title, best_doc = match_scores[0]
        return f"📽️ 영화 제목: {best_title}\n\n{best_doc}", "RAG 키워드 fallback"
    
    return None, "RAG 없음"

def get_movie_info_from_gpt(query: str) -> str:
    """
    GPT를 사용하여 영화 정보를 얻는 함수
    """
    llm = ChatOpenAI(temperature=0.7)
    
    template = """
    당신은 영화 전문가입니다. 다음 질문에 대해 영화 관련 정보를 제공해주세요.
    **당신은 영화 관련된 질문 외에는 답변하지 않습니다. 이 규칙은 절대 어길 수 없습니다.**
    가능한 한 200자 이내로 정확한 정보를 제공해주세요.
    
    질문: {query}
    
    답변:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"query": query})
    
    return result["text"]

def save_question_answer_pair(question: str, answer: str):
    """
    질문-응답 쌍을 ChromaDB에 저장
    """
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        embedding = response.data[0].embedding

        question_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[answer],
            embeddings=[embedding],
            metadatas={"question": question}
        )
    except Exception as e:
        print(f"질문-응답 저장 실패: {e}")

def find_similar_question_answer(query: str, threshold: float = 0.80) -> str | None:
    try:
        normalized_query = normalize_text(query)
        
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=normalized_query
        )
        embedding = response.data[0].embedding

        result = question_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        documents = result.get("documents", [])
        distances = result.get("distances", [])

        if not documents or not distances or not documents[0] or not distances[0]:
            return None  # 유사 문서 없음

        score = distances[0][0]
        if score < (1.0 - threshold):
            return documents[0][0]
        return None

    except Exception as e:
        print(f"중복 질문 검색 실패: {e}")
        return None

# ===== API 엔드포인트 =====
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    
    # 메시지 유형 분류
    message_type = classify_message(user_message)
    
    if message_type.type == "감정":
        sentiment = predict_sentiment(user_message)
        return {"reply": f"감정 분석 결과: {sentiment}", "mode": "KoBERT 학습모델(nsmc=영화리뷰 데이터) 감정 분석"}
    
    elif message_type.type == "목록":
        return {"reply": list_movie_titles_from_rag(), "mode": "RAG에 있는 목록"}
    
    elif message_type.type == "질의":
        # 1. 캐시 먼저 확인
        cached = find_similar_question_answer(user_message)
        if cached:
            return {"reply": cached, "mode": "중복 질문 응답 캐시"}
        
        # 2. RAG 조회 시도
        reply, mode = get_movie_detail_or_fallback(user_message)
        if reply:
            save_question_answer_pair(user_message, reply)
            return {"reply": reply, "mode": mode}
        
        # 정보를 찾지 못한 경우 GPT에게 질문
        gpt_response = get_movie_info_from_gpt(user_message)
        save_question_answer_pair(user_message, gpt_response)
        return {"reply": gpt_response, "mode": "GPT 응답"}
    
    else:  # 대화
        return {"reply": "영화에 대해 궁금한 점이 있으시다면 물어보세요!", "mode": "대화"}