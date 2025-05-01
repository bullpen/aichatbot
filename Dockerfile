# Python 3.10 기반 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 지정
WORKDIR /app

# 필요한 파일 복사
COPY server.py .  
COPY requirements.txt .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 서버 실행
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]
