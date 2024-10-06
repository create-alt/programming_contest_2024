import torch
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import json
import requests

from model import transition, SAC, Trainer 

SEED = 0
random.seed(SEED)
host_name = "http://localhost:8080"

"""
#本番前にはここを実行可能にして下の自作部分を隠す
# ヘッダーの設定
headers = {
    "Procon-Token": "osaka534508e81dbe6f70f9b2e07e61464780bd75646fdabf7b1e7d828d490e3"
}

# GET リクエストを送信
response = requests.get(host_name + "/problem", headers=headers)

# レスポンスのステータスコードを確認
if response.status_code == 200:

    data = response.json()
    # レスポンスの内容をファイルに保存
    #with open('response.json', 'w') as f:
    #    json.dump(response.json(), f, indent=4)
    print("Response caught and saved to response.json")

else:
    print(f"Failed to fetch data: {response.status_code}")

# 取得したデータを表示

start = data["board"]["start"]
goal  = data["board"]["goal"]

H, W = data["board"]["height"], data["board"]["width"]

for i in range(H):
    for j in range(W):
        board_train = int(start[i][j])
        board_test  = int(start[i][j])

        goal_train = int(goal[i][j])
        goal_test  = int(goal[i][j])


#定型抜き型を作成
cutter = [[[1]]]

for size in [2, 4, 8, 16, 32, 64, 128, 256]:
    grid = []
    #すべてが1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            grid[i].append(1)

    cutter.append(grid)

    grid = []
    #1マスおきに列が1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            if j%2 == 0:
                grid[i].append(1)
            else:
                grid[i].append(0)

    cutter.append(grid)

    grid = []
    #1マスおきに行が1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            if i%2 == 0:
                grid[i].append(1)
            else:
                grid[i].append(0)

    cutter.append(grid)

#cutterに一般抜き型を追加する
cutters = data["general"]["patterns"]

for _ in range(data["general"]["n"]):
    cut_info = cutters[i]
    grid=[]
    for i in range(cut_info["height"]):
        grid.append([])
        for j in range(cut_info["width"]):
            grid[i].append(int(cut_info["cells"][i][j]))
"""



"""## 環境のインスタンス化・学習・検証

### boardの作成と可視化
"""

"""
boardやcutterの作成
本来はjsonファイルの入力を受け取るが、プログラムのテスト用に直接作成
"""

#定型抜き型を作成
cutter = [[[1]]]

for size in [2, 4, 8, 16, 32, 64, 128, 256]:
    grid = []
    #すべてが1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            grid[i].append(1)

    cutter.append(grid)

    grid = []
    #1マスおきに列が1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            if j%2 == 0:
                grid[i].append(1)
            else:
                grid[i].append(0)

    cutter.append(grid)

    grid = []
    #1マスおきに行が1の抜き型
    for i in range(size):
        grid.append([])
        for j in range(size):
            if i%2 == 0:
                grid[i].append(1)
            else:
                grid[i].append(0)

    cutter.append(grid)

board_train = []

board_size = 32

for i in range(board_size):
  board_train.append([])
  for j in range(board_size):
      board_train[i].append(random.randint(0,3))

# goal_train = []

# for i in range(32):
#   goal_train.append([])
#   for j in range(32):
#       goal_train[i].append(random.randint(0,3))

goal_train = copy.deepcopy(board_train)
random.shuffle(goal_train)

board_test = copy.deepcopy(board_train)
goal_test = copy.deepcopy(goal_train)

count = 0
for i in range(board_size):
  for j in range(board_size):
    if(board_test[i][j] == goal_test[i][j]):
      count+=1

print(count)


"""
boardを可視化
"""
# # 0~3の値に対応する色を定義
# cmap = ListedColormap(['red', 'green', 'blue', 'yellow'])

# # 図を描画
# plt.imshow(board_test, cmap=cmap, interpolation='none')

# # カラーバーを表示して、各色が何の値に対応するかを表示
# cbar = plt.colorbar(ticks=[0, 1, 2, 3])
# cbar.ax.set_yticklabels(['0', '1', '2', '3'])  # ラベルを設定

# # グリッド線を追加
# plt.grid(False)  # グリッドを非表示にする場合はTrueをFalseに変更

# # 図を表示
# plt.show()

# # 0~3の値に対応する色を定義
# cmap = ListedColormap(['red', 'green', 'blue', 'yellow'])

# # 図を描画
# plt.imshow(goal_test, cmap=cmap, interpolation='none')

# # カラーバーを表示して、各色が何の値に対応するかを表示
# cbar = plt.colorbar(ticks=[0, 1, 2, 3])
# cbar.ax.set_yticklabels(['0', '1', '2', '3'])  # ラベルを設定

# # グリッド線を追加
# plt.grid(False)  # グリッドを非表示にする場合はTrueをFalseに変更

# # 図を表示
# plt.show()

"""### 学習の開始"""

"""
環境とネットワークのインスタンス生成と訓練の開始
"""
SEED = 0
REWARD_SCALE = 0.99
NUM_STEPS = 5 * 10 ** 4
BATCH_SIZE = 1000
EVAL_INTERVAL = BATCH_SIZE * 10

#以下の引数は学習・テストデータであり、別で作成・形成を行う
env = transition(board_train, cutter, goal_train, EPISODE_SIZE=BATCH_SIZE)
env_test = transition(board_test, cutter, goal_test, test = True, EPISODE_SIZE=BATCH_SIZE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

algo = SAC(
    state_shape=env.state_shape, #実験時にはそれに合わせた値で、本制作時には最大値で設定・学習する
    action_shape=env.action_shape,     #上同
    num_of_cutter = env.num_of_cutter,
    device=device,
    seed=SEED,
    reward_scale=REWARD_SCALE,
    batch_size=BATCH_SIZE,
    lr_actor=3e-4,
    lr_critic=3e-3,
    replay_size=10**3,
    start_steps=BATCH_SIZE,
    # pretrain = True,
    # model_weight_name = 'model_0.40625'
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()

"""
コーディングメモ

ある程度までそろえられてもそれ以降決め手がないので、指針となるような爆発的な報酬が欲しいところ
一方で、負の報酬についてもうまく導入して、余計な行動を阻害することも考える
ー＞微小な変化に対応できず、argmaxをとったときに同じ行動を返す場合に負の報酬など
    （実際の行動はrandom.choiceでよいが、報酬自体はそれで与える）

ネットワークに入力を通して出力をもとにbackwardを行うとき、
道中でtensorを定義しなおしたりnumpyに型変換したり、detachやno_gradなどを使用すると勾配が伝わらなくなり、学習ができなくなるので注意
"""


"""### 報酬と行動回数の変化の可視化"""

#最大報酬の変化を可視化
trainer.plot_return()

#最大報酬獲得のstep数を可視化
trainer.plot_steps()