import pandas as pd
import pickle
from openai.embeddings_utils import (
    get_embedding
)

# Retorna o embedding de uma string usando o cache para evitar computar redundâncias
def embedding_from_string(
    string: str,
    model: str,
    embedding_cache,
    embedding_cache_path
) -> list:
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


# Retorna o arquivo de cache com a lista de embeddings já extraídos
def pegar_embedding_cache(embedding_cache_path):    
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
        return embedding_cache
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache


# Adiciona uma coluna de embeddings a um data frame
def adicionar_embeddings(df, nome_coluna, model, embedding_cache, embedding_cache_path):
    embeddings = []
    for valor in df[nome_coluna]:
        embedding = embedding_from_string(str(valor), 
                                          model=model,
                                          embedding_cache=embedding_cache,
                                          embedding_cache_path=embedding_cache_path
                                          )
        embeddings.append(embedding)
    df['Embeddings'] = embeddings
    return df