import pandas as pd

# 使用绝对路径读取 Parquet 文件
"""
documents.parquet 文件包含完整的 source year month day id title creation metadata 
"""
df_documents = pd.read_parquet('E:\\AI\\code\\opensource\\graphrag\\output\\documents.parquet')  # 替换为实际路径

"""
 包含 month text id document_ids n_tokens
 month 是因为 配置文件配置了 group_by_columns
"""
# df_text_units = pd.read_parquet('E:\\AI\\code\\opensource\\graphrag\\output\\text_units.parquet')  # 替换为实际路径
# print(df_documents.head(5))

print("===============================================")

# print(df_text_units.head(5))

print("===============================================")
from tabulate import tabulate

# 其中，headers='keys' 表示使用列名作为表头，tablefmt='pretty' 表示使用 pretty 格式化输出，showindex=False 表示不显示行索引，stralign='left' 表示左对齐，maxcolwidths=[20, 20, 20, 20, 20] 表示每列的最大宽度为20
print(tabulate(df_documents, headers='keys', tablefmt='pretty', showindex=False, stralign='left', maxcolwidths=[20, 20, 20, 20, 20]))

print("===============================================")

# 其中，headers='keys' 表示使用列名作为表头，tablefmt='pretty' 表示使用 pretty 格式化输出，showindex=False 表示不显示行索引，stralign='left' 表示左对齐，maxcolwidths=[20, 20, 20, 20, 20] 表示每列的最大宽度为20
# print(tabulate(df_text_units, headers='keys', tablefmt='pretty', showindex=False, stralign='left', maxcolwidths=[20, 20, 20, 20, 20]))