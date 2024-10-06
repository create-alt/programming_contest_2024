import json

# response.json ファイルを読み込む
with open('response.json', 'r') as f:
    data = json.load(f)

# 取得したデータを表示
print(data)