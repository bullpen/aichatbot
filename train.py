import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from kobert_transformers import get_tokenizer
from transformers import BertForSequenceClassification
import torch
import os
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
import dotenv
from kobert_transformers import get_tokenizer

dotenv.load_dotenv()

print(f"Test: {os.getenv('TRAIN_FILE_PATH')}")
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """학습 설정을 관리하는 데이터 클래스"""
    output_dir: str = "./results" # 모델 저장 경로
    eval_strategy: str = "epoch" # 평가 전략
    learning_rate: float = 2e-5 # 학습률
    train_batch_size: int = 16 # 학습 배치 크기
    eval_batch_size: int = 16 # 평가 배치 크기
    num_epochs: int = 5 # 학습 에폭 수(에폭이란 데이터셋을 한 번 전체 훑는 것을 의미)
    weight_decay: float = 0.01 # 가중치 감소 값
    fp16: bool = True # 16비트 연산 사용 여부

class SentimentTrainer:
    """감성 분석 모델 학습을 위한 클래스"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """학습기 초기화
        
        Args:
            config: 학습 설정 (기본값 사용 시 None)
        """
        self.config = config or TrainingConfig()
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        
    def _setup_device(self) -> torch.device:
        """GPU 사용 가능 여부 확인 및 디바이스 설정"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """NSMC 데이터 로드 및 전처리
        
        Returns:
            학습 데이터와 테스트 데이터 DataFrame 튜플
        """
        try:
            train_file_path = os.getenv("TRAIN_FILE_PATH")
            test_file_path = os.getenv("TEST_FILE_PATH")

            print(f"train_file_path: {train_file_path}, test_file_path: {test_file_path}")
            
            if not train_file_path or not test_file_path:
                raise ValueError("TRAIN_FILE_PATH 또는 TEST_FILE_PATH 환경 변수가 설정되지 않았습니다")
            
            # 데이터 로드
            train_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
            test_df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
            
            # 데이터 전처리
            train_df = self._preprocess_data(train_df)
            test_df = self._preprocess_data(test_df)
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 수행
        
        Args:
            df: 전처리할 DataFrame
            
        Returns:
            전처리된 DataFrame
        """
        # 불필요한 컬럼 제거
        df = df.drop(columns=['id'])
        # null 값이 있는 행 제거
        df = df.dropna(subset=['document'])
        return df
    
    def _setup_model(self) -> None:
        """KoBERT 모델과 토크나이저 설정"""
        try:
            
            checkpoint_path = "D:/aiproject/results/checkpoint-46875"

            self.tokenizer = get_tokenizer()
            # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = BertForSequenceClassification.from_pretrained(
                checkpoint_path,
                num_labels=2
            )
            self.model.to(self.device)
            logger.info("모델 및 토크나이저 설정 완료")
        except Exception as e:
            logger.error(f"모델 설정 실패: {e}")
            raise
    
    def _tokenize_function(self, examples: dict) -> dict:
        """토크나이징 함수
        
        Args:
            examples: 토크나이징할 예제 딕셔너리
            
        Returns:
            토크나이징된 결과 딕셔너리
        """
        return self.tokenizer(
            examples['document'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    def _prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """데이터셋 준비
        
        Args:
            train_df: 학습 데이터 DataFrame
            test_df: 테스트 데이터 DataFrame
        """
        try:
            # DataFrame을 Dataset으로 변환 (인덱스 보존하지 않음)
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
            test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
            
            # 토크나이징 적용
            train_dataset = train_dataset.map(self._tokenize_function, batched=True)
            test_dataset = test_dataset.map(self._tokenize_function, batched=True)
            
            # 불필요한 컬럼 제거 (document만 제거)
            train_dataset = train_dataset.remove_columns(['document'])
            test_dataset = test_dataset.remove_columns(['document'])
            
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            logger.info("데이터셋 준비 완료")
            
        except Exception as e:
            logger.error(f"데이터셋 준비 실패: {e}")
            raise
    
    def _setup_training_args(self) -> TrainingArguments:
        """학습 인자 설정
        
        Returns:
            설정된 TrainingArguments 객체
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy=self.config.eval_strategy,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16
        )
    
    def train(self) -> None:
        """모델 학습 실행"""
        try:
            # 데이터 로드
            train_df, test_df = self._load_data()
            
            # 모델 설정
            self._setup_model()
            
            # 데이터셋 준비
            self._prepare_datasets(train_df, test_df)
            
            # 학습 인자 설정
            training_args = self._setup_training_args()
            
            # Trainer 설정
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset
            )
            
            # 학습 시작
            logger.info("학습 시작")
            trainer.train()
            
            # 모델 저장
            # self.tokenizer.save_pretrained(self.config.output_dir)
            
            # 모델 저장
            try:
                # 먼저 모델 저장
                self.model.save_pretrained(self.config.output_dir)

                # KoBERT 토크나이저는 save_vocabulary()를 직접 써야 함
                vocab_path = self.tokenizer.save_vocabulary(self.config.output_dir)[0]  # filename_prefix 없이 저장
                logger.info(f"Vocab 파일 저장 위치: {vocab_path}")

                # 토크나이저 config는 수동 저장 (필요할 경우)
                tokenizer_config_path = os.path.join(self.config.output_dir, "tokenizer_config.json")
                with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                    f.write('{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}')

                logger.info(f"모델과 토크나이저가 {self.config.output_dir}에 저장되었습니다")

            except Exception as e:
                logger.error(f"모델 저장 중 오류 발생: {e}")
            
            
            logger.info(f"모델이 {self.config.output_dir}에 저장되었습니다")
            
        except Exception as e:
            logger.error(f"학습 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    try:
        trainer = SentimentTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        raise

if __name__ == "__main__":
    main()