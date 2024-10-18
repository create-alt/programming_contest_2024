"""
実行時にはカレントディレクトリをrunsにしてから実行してください！重みが保存できなくなります。
"""

import torch
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import json
import requests

from model import SAC, Trainer
from Env import transition
from create_board import create_train_board

SEED = 0
random.seed(SEED)
host_name = "http://localhost:8080"

SEED = 20
REWARD_SCALE = 0.9
NUM_STEPS = 10 ** 5
BATCH_SIZE = 200
EVAL_INTERVAL = BATCH_SIZE * 10

random.seed(2)
goal_board = []
board_shape = [128, 128]
for i in range(board_shape[0]):
  goal_board.append([])

  for j in range(board_shape[1]):
    goal_board[i].append(random.randint(0, 3))


board_train, goal_train, cutter, get_actions = create_train_board(seed=200,
                                                                  board_shape=board_shape,
                                                                  cutter_add_num=0,
                                                                  num_of_shuffle=BATCH_SIZE,  # 最短何手でgoalにたどり着くのかを指定
                                                                  goal=goal_board)

board_test = copy.deepcopy(board_train)
goal_test = copy.deepcopy(goal_train)

count = 0
for i in range(len(board_train)):
  for j in range(len(board_train[0])):
    if (board_test[i][j] == goal_test[i][j]):
      count += 1

print(count)

# 以下の引数は学習・テストデータであり、別で作成・形成を行う
env = transition(board_train, cutter, goal_train,
                 EPISODE_SIZE=BATCH_SIZE, get_actions=get_actions)
env_test = transition(board_test, cutter, goal_test, test=True,
                      EPISODE_SIZE=BATCH_SIZE, get_actions=get_actions)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

algo = SAC(
    state_shape=env.state_shape,
    action_shape=env.action_shape,
    num_of_cutter=env.num_of_cutter,
    device=device,
    seed=SEED,
    reward_scale=REWARD_SCALE,
    batch_size=BATCH_SIZE,
    lr_actor=5e-4,
    lr_critic=5e-4,
    replay_size=4 * 10**3,
    start_steps=BATCH_SIZE,
    tau=1e-3,
    # pretrain = True,
    # model_weight_name = 'model_592,2048'
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

# 最大報酬の変化を可視化
trainer.plot_return()

# 最大報酬獲得のstep数を可視化
trainer.plot_steps()
