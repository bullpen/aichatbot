from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()
print(tokenizer.tokenize("안녕하세요. 반갑습니다."))