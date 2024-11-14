# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.spatial.distance import cosine
# from nltk.corpus import stopwords
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import torch

# def improved_similarity(text1, text2):
#     # 去除停用词
#     stop_words = set(stopwords.words('english'))
    
#     # 使用TF-IDF
#     tfidf = TfidfVectorizer(stop_words=stop_words)
#     vectors = tfidf.fit_transform([text1, text2])
    
#     # 计算余弦相似度
#     similarity = 1 - cosine(vectors[0].toarray(), vectors[1].toarray())
    
#     return similarity

# # 使用sentence-transformers
# from sentence_transformers import SentenceTransformer

# def bert_similarity(text1, text2):
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
#     # 获取句子向量
#     embedding1 = model.encode(text1)
#     embedding2 = model.encode(text2)
    
#     # 计算余弦相似度
#     similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
#     return similarity

# def alternative_bert_similarity(text1, text2):
#     try:
#         # 加载小型模型
#         tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
#         model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        
#         # 编码文本
#         inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=128)
#         inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
#         # 获取embeddings
#         with torch.no_grad():
#             outputs1 = model(**inputs1)
#             outputs2 = model(**inputs2)
        
#         # 使用CLS token的输出
#         emb1 = outputs1.last_hidden_state[:, 0, :].numpy()
#         emb2 = outputs2.last_hidden_state[:, 0, :].numpy()
        
#         # 计算相似度
#         similarity = np.dot(emb1.flatten(), emb2.flatten()) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
#         return similarity
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         return None


# if __name__=="__main__":
#     # text1='israel and wars'
#     # text2='israel and missille'
#     # # text1 = "The company's revenue increased by 10%."
#     # # text2 = "The firm's sales grew by 10 percent."
#     # #s1 = improved_similarity(text1, text2)
#     # #s2 = alternative_bert_similarity(text1, text2)
#     # s1 = bert_similarity(text1, text2)
#     # print(f's1:{s1}')
#     # s2 = alternative_bert_similarity(text1, text2)
#     # print(f's2:{s2}')

from sentence_transformers import SentenceTransformer
import os

def check_model_cache():
    # 获取默认缓存目录
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'sentence_transformers')
    
    # 打印缓存目录
    print(f"Default cache directory: {cache_dir}")
    
    # 列出缓存的模型
    if os.path.exists(cache_dir):
        models = os.listdir(cache_dir)
        print("\nCached models:")
        for model in models:
            model_path = os.path.join(cache_dir, model)
            size = sum(os.path.getsize(os.path.join(dirpath,filename)) 
                      for dirpath, _, filenames in os.walk(model_path)
                      for filename in filenames)
            print(f"- {model}: {size/1024/1024:.2f} MB")
    
    return cache_dir

# 使用自定义缓存目录
def load_model_custom_path(model_name, custom_path):
    try:
        # 加载模型到指定位置
        model = SentenceTransformer(model_name, cache_folder=custom_path)
        print(f"Model loaded successfully to: {custom_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# 示例使用
if __name__ == "__main__":
    # 查看当前缓存
    cache_dir = check_model_cache()
    
    # 使用自定义路径（可选）
    custom_path = "./my_models"
    model = load_model_custom_path('paraphrase-MiniLM-L6-v2', custom_path)
    
    # 测试模型
    if model:
        sentences = ['This is a test sentence', 'Another test sentence']
        embeddings = model.encode(sentences)
        print(f"\nModel test successful. Embedding shape: {embeddings.shape}")