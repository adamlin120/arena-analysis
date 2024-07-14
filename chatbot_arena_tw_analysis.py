from datasets import load_dataset


def main():
    ds = load_dataset("aigrant/tw_chatbot_arena", "all", split='train')
    """
    In [4]: ds
Out[4]: 
Dataset({
    features: ['question_id', 'model_a', 'model_b', 'winner', 'judge', 'conversation_a', 'conversation_b', 'turn', 'anony', 'language', 'tstamp'],
    num_rows: 3436
})

In [5]: ds[0]
Out[5]: 
{'question_id': '7d9cfd9e501b4428a054daaa40571dd3',
 'model_a': '### Model A: gpt-3.5-turbo',
 'model_b': '### Model B: Llama-3-Taiwan-70B-Instruct',
 'winner': 'tie (bothbad)',
 'judge': 'arena_user_2001:b400:e270:382c:f898:d8b3:ef52:7f1a',
 'conversation_a': [{'content': '中國是什麼的縮寫', 'num_tokens': 15, 'role': 'user'},
  {'content': '中國的縮寫是「CN」。', 'num_tokens': 14, 'role': 'assistant'},
  {'content': '中國國慶日的日期是？', 'num_tokens': 13, 'role': 'user'},
  {'content': '中國國慶日是每年的10月1日。', 'num_tokens': 18, 'role': 'assistant'},
  {'content': '中國現任總統是誰？', 'num_tokens': 15, 'role': 'user'},
  {'content': '中國的現任總統是習近平。', 'num_tokens': 19, 'role': 'assistant'},
  {'content': '中華民國是否為中國', 'num_tokens': 13, 'role': 'user'},
  {'content': '中華民國（台灣）和中華人民共和國（中國大陸）是兩個主權國家，都自稱為「中國」，但政治上並不相互承認對方的主權。因此，中華民國和中華人民共和國在國際上被視為兩個不同的國家。',
   'num_tokens': 119,
   'role': 'assistant'},
  {'content': '中國的總統分別是誰', 'num_tokens': 15, 'role': 'user'},
  {'content': '中國大陸的總統是習近平，而台灣的總統是蔡英文。',
   'num_tokens': 36,
   'role': 'assistant'},
  {'content': '賴清德的現職是？', 'num_tokens': 12, 'role': 'user'},
  {'content': '賴清德是台灣的前行政院長，目前則擔任民進黨主席。',
   'num_tokens': 34,
   'role': 'assistant'}],
 'conversation_b': [{'content': '中國是什麼的縮寫', 'num_tokens': 15, 'role': 'user'},
  {'content': '中國的縮寫是中國。', 'num_tokens': 14, 'role': 'assistant'},
  {'content': '中國國慶日的日期是？', 'num_tokens': 13, 'role': 'user'},
  {'content': '中國的國慶日是10月1日。', 'num_tokens': 16, 'role': 'assistant'},
  {'content': '中國現任總統是誰？', 'num_tokens': 15, 'role': 'user'},
  {'content': '中國的現任總統是習近平。', 'num_tokens': 19, 'role': 'assistant'},
  {'content': '中華民國是否為中國', 'num_tokens': 13, 'role': 'user'},
  {'content': "中華民國（Republic of China，簡稱ROC）和中華人民共和國（People's Republic of China，簡稱PRC）是兩個不同的政治實體。\n\n中華民國成立於1912年，最初管轄整個中國大陸。1949年，因第二次國共內戰，中華民國政府遷往台灣，實際管轄範圍限縮至台澎金馬等地區。\n\n中華人民共和國於1949年10月1日成立，實際管轄中國大陸和香港、澳門等地區。\n\n目前，在國際上大多數國家承認中華人民共和國為中國的唯一合法代表。中華民國僅與少數國家維持正式外交關係。\n\n因此，中華民國和中華人民共和國分別是兩個不同的政權，各自宣稱代表中國，但實際控制的領土不同。國際社會普遍認知的「中國」主要指中華人民共和國。",
   'num_tokens': 366,
   'role': 'assistant'},
  {'content': '中國的總統分別是誰', 'num_tokens': 15, 'role': 'user'},
  {'content': '中華人民共和國（PRC）和中華民國（ROC）是兩個不同的政治實體，各自有自己的國家元首。\n\n中華人民共和國的國家元首是習近平，他同時擔任中國共產黨中央委員會總書記、中華人民共和國主席、中央軍事委員會主席等職務。中華人民共和國沒有總統一職，其國家元首為「國家主席」。\n\n中華民國（台灣）的國家元首是總統，現任總統為蔡英文。她於2016年當選中華民國第14任總統，並於2020年連任。\n\n需要注意的是，中華人民共和國認為台灣是其領土的一部分，並宣稱對整個中國擁有主權。然而，中華民國則認為自己是主權獨立的國家，並稱中華人民共和國為「大陸」或「中國大陸」。',
   'num_tokens': 374,
   'role': 'assistant'},
  {'content': '賴清德的現職是？', 'num_tokens': 12, 'role': 'user'},
  {'content': '賴清德是中華民國（台灣）的政治人物，目前擔任中華民國副總統。他於2020年與蔡英文搭檔參選並當選副總統，於同年5月20日就職。\n\n在擔任副總統之前，賴清德曾擔任過台南市市長、行政院院長等重要職務。他也是民主進步黨的重要成員之一。\n\n作為副總統，賴清德在台灣的政治舞台上扮演著重要的角色，協助總統處理國家事務，並代表政府出席重要活動和儀式。',
   'num_tokens': 240,
   'role': 'assistant'}],
 'turn': 6,
 'anony': False,
 'language': 'Chinese',
 'tstamp': 1720677863.9913}
    """

    from src.text_clustering import ClusterClassifier
    import pandas as pd
    import torch

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Extract user conversations from the dataset
    user_conversations = []
    for item in ds:
        conversation = ' '.join([turn['content'] for turn in item['conversation_a'] if turn['role'] == 'user'])
        user_conversations.append(conversation)
    # Initialize and fit the ClusterClassifier
    cc = ClusterClassifier(
        embed_model_name="BAAI/bge-m3",  # Using BAAI/bge-m3 for embedding
        embed_device=device,  # Use CUDA if available
        dbscan_min_samples=10,
        summary_create=True,
        topic_mode="multiple_topics",
        summary_instruction="用三個詞或短語總結這些台灣用戶問題的主要主題，以逗號分隔。請直接產生總結。"
    )
    
    embeddings, labels, summaries = cc.fit(user_conversations)
    
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        'text': user_conversations,
        'cluster': labels
    })
    
    # Print cluster summaries
    print("User Question Cluster Summaries:")
    for cluster, summary in summaries.items():
        if cluster != -1:
            print(f"Cluster {cluster}: {summary}")
    
    # Show the clustering results
    cc.show()
    
    # Save the results
    cc.save("chatbot_arena_user_clustering_results")
    
    print("User question clustering analysis complete. Results saved in 'chatbot_arena_user_clustering_results' directory.")

if __name__ == "__main__":
    main()