from chromadb import HttpClient
from openai import OpenAI
import uuid
import json
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MovieData:
    """영화 데이터를 표현하는 데이터 클래스"""
    content: str  # 영화 내용
    metadata: Dict[str, Any]  # 영화 메타데이터
    doc_id: str  # 문서 ID

class MovieEmbeddingService:
    """영화 임베딩 및 ChromaDB 작업을 처리하는 서비스 클래스"""
    
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        """임베딩 서비스 초기화
        
        Args:
            chroma_host: ChromaDB 서버 호스트 주소
            chroma_port: ChromaDB 서버 포트 번호
        """
        self.chroma_client = HttpClient(host=chroma_host, port=chroma_port)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection_name = "movie-collection"
        self.embedding_model = "text-embedding-ada-002"
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

    def _delete_existing_collection(self) -> None:
        """기존 컬렉션 삭제"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info("기존 컬렉션이 성공적으로 삭제되었습니다")
        except Exception as e:
            logger.warning(f"컬렉션 삭제 실패 또는 컬렉션이 존재하지 않음: {e}")

    def _get_or_create_collection(self):
        """새 컬렉션 생성 또는 기존 컬렉션 가져오기"""
        return self.chroma_client.get_or_create_collection(name=self.collection_name)

    def _load_movie_data(self, file_path: str) -> List[Dict[str, Any]]:
        """JSON 파일에서 영화 데이터 로드
        
        Args:
            file_path: 영화 데이터가 포함된 JSON 파일 경로
            
        Returns:
            영화 데이터 딕셔너리 리스트
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"영화 데이터 로드 실패: {e}")
            raise

    def _process_movie_data(self, item: Dict[str, Any]) -> Optional[MovieData]:
        """개별 영화 데이터 처리
        
        Args:
            item: 영화 데이터가 포함된 딕셔너리
            
        Returns:
            처리된 MovieData 객체 또는 유효하지 않은 경우 None
        """
        content = item.get("content", "")
        if not content:
            return None

        metadata = item.get("metadata", {})
        
        # keywords가 JSON 문자열인지 확인
        if "keywords" in metadata and isinstance(metadata["keywords"], list):
            metadata["keywords"] = json.dumps(metadata["keywords"])
        elif "keywords" not in metadata:
            metadata["keywords"] = "[]"

        return MovieData(
            content=content,
            metadata=metadata,
            doc_id=str(uuid.uuid4())
        )

    def _get_embedding(self, text: str) -> List[float]:
        """OpenAI API를 사용하여 텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 값 리스트
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise

    def process_movies(self, json_file_path: str) -> None:
        """ChromaDB에 영화 데이터 처리 및 저장
        
        Args:
            json_file_path: 영화 데이터가 포함된 JSON 파일 경로
        """
        try:
            # 컬렉션 초기화
            self._delete_existing_collection()
            collection = self._get_or_create_collection()
            
            # 영화 데이터 로드 및 처리
            movie_data = self._load_movie_data(json_file_path)
            
            for item in movie_data:
                processed_movie = self._process_movie_data(item)
                if not processed_movie:
                    continue
                
                logger.info(f"처리 중인 영화: {processed_movie.metadata.get('title')} | 키워드: {processed_movie.metadata.get('keywords')}")
                
                embedding = self._get_embedding(processed_movie.content)
                
                collection.add(
                    ids=[processed_movie.doc_id],
                    documents=[processed_movie.content],
                    embeddings=[embedding],
                    metadatas=[processed_movie.metadata]
                )
            
            logger.info("모든 영화 데이터가 ChromaDB에 성공적으로 처리 및 저장되었습니다")
            
        except Exception as e:
            logger.error(f"영화 처리 실패: {e}")
            raise

def main():
    """영화 임베딩 프로세스를 실행하는 메인 함수"""
    try:
        service = MovieEmbeddingService()
        service.process_movies("newmovies.json")
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        raise

if __name__ == "__main__":
    main()
