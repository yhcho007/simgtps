from pymilvus import connections, Collection

def connect_milvus(host='localhost', port=19530):
    connections.connect('default', host=host, port=port)
    print('Connected to Milvus')
