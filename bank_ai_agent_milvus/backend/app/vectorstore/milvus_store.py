"""
Milvus 기반 Vector Store Client
---------------------------------
- pymilvus 라이브러리를 사용하여 벡터 임베딩을 저장/검색합니다.
- 기존 FAISS 구현을 대체합니다.
- 컬렉션 구조: id(str), text(str), embedding(float vector)
"""
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

class MilvusClient:
    def __init__(self, host="localhost", port="19530", collection_name="bank_docs"):
        # Milvus 연결
        connections.connect(alias="default", host=host, port=port)
        self.collection_name = collection_name

        # 컬렉션 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, description="Bank document embeddings")

        # 컬렉션 생성 (존재하지 않으면)
        if self.collection_name not in [c.name for c in Collection.list_collections()]:
            self.collection = Collection(name=self.collection_name, schema=schema)
            # IVF_FLAT index 예시
            self.collection.create_index(
                field_name="embedding",
                index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
            )
        else:
            self.collection = Collection(self.collection_name)

    def add(self, ids, texts, embeddings):
        """데이터 삽입: id, text, embedding 리스트를 전달"""
        entities = [ids, texts, embeddings]
        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_embedding, top_k=5):
        """임베딩 기반 검색 수행"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text"]
        )

        hits = []
        for res in results[0]:
            hits.append({
                "id": res.entity.get("id"),
                "text": res.entity.get("text"),
                "score": res.distance
            })
        return hits
