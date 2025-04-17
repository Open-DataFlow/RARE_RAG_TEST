from clickhouse_connect import get_client

# 连接参数（根据场景选择）
config = {
    'host': 'localhost',
    'port': 8123,
    'username': 'default',
    'password': '',
    'database': 'default',
}

# 创建客户端
client = get_client(**config)

# 查询rag_vectors表的前100条数据
result = client.query("SELECT * FROM rag_vectors LIMIT 10")
print("\nFirst 100 rows from rag_vectors table:")
for row in result.named_results():
    print(row)

# 2. 删除表中的所有数据
client.command("TRUNCATE TABLE rag_vectors")
print("\nAll data in rag_vectors table has been deleted.")

# 3. 验证表是否为空
result = client.query("SELECT count() FROM rag_vectors")
print(f"\nAfter deletion - Number of rows in rag_vectors: {result.first_row[0]}")
# 关闭连接
client.close()