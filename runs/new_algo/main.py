import requests
import subprocess
import json

host_name = "http://localhost:8080"

token = "token1"

# ヘッダーの設定
headers = {
    "Procon-Token": token
}

# GET リクエストを送信
response = requests.get(host_name + "/problem", headers=headers)

# レスポンスのステータスコードを確認
if response.status_code == 200:

  data = response.json()

  # レスポンスの内容をファイルに保存
  with open('./response.json', 'w') as f:
    json.dump(response.json(), f, indent=4)

  print("Response caught and saved to response.json")

else:
  print(f"Failed to fetch data: {response.status_code}")


# C++プログラムの実行ファイルを指定して実行
result = subprocess.run(["./test.exe"], capture_output=True, text=True)

# 実行結果を表示
print(result.stdout)

json_name = "./solution.json"

with open(json_name, 'r') as f:
  solution_data = json.load(f)


# ヘッダーの設定
headers = {"Content-Type": "application/json",
           "Procon-Token": token}

# POST リクエストを送信
response = requests.post(
    host_name + "/answer", headers=headers, json=solution_data)

# レスポンスのステータスコードと内容を確認
print("Status Code:", response.status_code)
