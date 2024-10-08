from abc import ABC, abstractmethod
from time import time
from datetime import timedelta
import random
import math
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gc
from pprint import pprint
import json
import requests

import torch
import torchvision
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from torchvision import models


path = "../model_weights/"
host_name = "http://localhost:8080"

"""## 関数定義"""

def mySqueeze(x):
    """Optimize mySqueeze by avoiding unnecessary list conversion."""
    return np.squeeze(np.array(x)).tolist()


def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis

def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    """ 平均(mean)，標準偏差の対数(log_stds)でパラメータ化した方策における，行動(actions)の確率密度の対数を計算する． """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

"""## 環境定義"""

#2重ループが頻出しており、実行時間が長いと考えられるので要改善
class transition():
    """
    本クラスはenv(学習環境)として扱うクラスであるので、
    行動に対して次の状態と報酬、実行が終わったかどうかを返す。
    """
    def __init__(self, board, cutter, goal, frequ=1, test=False, EPISODE_SIZE = 1000):
        """
        board (2次元list)  => 最初に与えられたボードの状態(0~3)
                                臨時的な値として'-1'は穴抜けを表す

        cutter (3次元list) => 今回のboardに対して使用可能な抜き型を格納した配列
                              0~24番まではどの場合でも固定

        goal(2次元list)    => ボードの理想形(目標),boardとサイズも含んでいる要素数も等しい

        frequ(整数)        => 報酬の獲得頻度
        """
        self.start = copy.deepcopy(board) #startは初期値として固定しておく。boardは可変
        self.board = copy.deepcopy(board)
        self.x_boardsize, self.y_boardsize = len(board), len(board[0])
        self.goal  = goal
        self.cutter = cutter
        self.done  = False
        self.num_step = 0
        self._max_episode_steps = EPISODE_SIZE
        self.frequ = frequ #frequは報酬の獲得頻度を表す。
        self.before_rew = 0

        self.rew   = 0

        self.h_countforrew, self.w_countforrew= [], [] # 左or上詰めで、行・列の一致に報酬を与えるが、もともと一致していると過剰な報酬を与えることになるので、事前に一致している行・列を格納

        self.default_eq = 0
        for i in range(self.x_boardsize):
          for j in range(self.y_boardsize):
            if board[i][j] == goal[i][j]:
              self.default_eq += 1

        self.before_eq = self.default_eq / len(board) * len(board[0])

        self.num_of_cutter = len(self.cutter)
        self.state_shape  = [len(board), len(board[0])]
        self.action_shape = [4]

        self.max_rew = -10000
        self.max_eq  = -1
        self.num_eq = 0
        self.best_step = 0

        self.ans = {"n":0, "ops":[]}

        self.ans_board = None

        self.test = test
        self.before_act = None

    def step(self, act):
        """
        本メソッドでは、別クラスで出力したactをもとに本環境の更新(step)を行い、報酬や状態などを返す。
        """
        self.num_step += 1

        state = copy.deepcopy(self.board)

        #actは別クラスで出力し、本メソッドに入力する
        next_state = self.action(state,[act[0], act[1]], self.cutter[act[2]], act[3])
        self.board = copy.deepcopy(next_state)

        if self.num_step % self.frequ == 0:
            self.rew += self.reward(state,next_state, act)
            """
            max_rewの時までの行動とその時の状態を保存して出力するように書き換え
            """

        #目標形にたどり着いた場合行動を終了する。
        if self.board == self.goal:
            self.done = True

        if self.num_step == self._max_episode_steps:
            self.done = True

        if self.test:
            # act[0] = (act[0] * self.x_boardsize) // 256
            # act[1] = (act[1] * self.y_boardsize) // 256
            self.ans["ops"].append({"p":act[2], "x":act[0], "y":act[1], "s":act[3]})

        self.before_act = copy.deepcopy(act)

        #next_state : 次の状態, self.rew : 今回の行動への報酬, self.done : 行動が終了するかどうか
        return next_state, self.rew, self.done, self.num_step

    def action(self, board, pos, cutter, direction, batch_size=1):
      """
      本メソッドでは、選択した行動の結果から次の状態を計算する
      """
      start = time()

      # cutterのサイズと型抜き座標を定数として格納
      X_SIZE, Y_SIZE = len(cutter), len(cutter[0])
      X, Y = pos[0], pos[1]
      x_boardsize, y_boardsize = len(board), len(board[0])

      # 移動方向をベクトルで定義
      direction_vectors = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
      dx, dy = direction_vectors[direction]

      # cut_valを動的に確保 (メモリの無駄を避けるため、必要サイズだけ確保)
      cut_val = []

      # 抜きピースを保存し、穴を空ける
      for i in range(X_SIZE):
          for j in range(Y_SIZE):
              if 0 <= X + i < x_boardsize and 0 <= Y + j < y_boardsize and cutter[i][j] == 1:
                  cut_val.append(board[X + i][Y + j])
                  board[X + i][Y + j] = -1

      #direction (int)  => 型抜き後移動する方向(0:上 1:下 2:左 3:右)

      # board全体の走査を省略するため、カッターの影響を受けた範囲のみ移動処理を実施
      if direction in [0, 1]:  # 上下方向の移動
          for j in range(Y, Y + Y_SIZE):
              if 0 <= j < y_boardsize:
                  count = 0
                  if direction == 0:  # 上方向
                      for i in range(X, x_boardsize):
                          if board[i][j] == -1:
                              count += 1
                          elif count > 0:
                              board[i - count][j] = board[i][j]
                              board[i][j] = -1
                  else:  # 下方向
                      for i in range(min(X + X_SIZE - 1, x_boardsize - 1), -1, -1):
                          if board[i][j] == -1:
                              count += 1
                          elif count > 0:
                              board[i + count][j] = board[i][j]
                              board[i][j] = -1

      elif direction in [2, 3]:  # 左右方向の移動
          for i in range(X, X + X_SIZE):
              if 0 <= i < x_boardsize:
                  count = 0
                  if direction == 2:  # 左方向
                      for j in range(Y, y_boardsize):
                          if board[i][j] == -1:
                              count += 1
                          elif count > 0:
                              board[i][j - count] = board[i][j]
                              board[i][j] = -1
                  else:  # 右方向
                      for j in range(min(Y + Y_SIZE - 1, y_boardsize - 1), -1 , -1):
                          if board[i][j] == -1:
                              count += 1
                          elif count > 0:
                              board[i][j + count] = board[i][j]
                              board[i][j] = -1

      # 元の位置にcut_valを戻す (全探索を避け、ピースが置かれた場所にのみ戻す)
      cut_index = 0
      for i in range(X_SIZE):
          for j in range(Y_SIZE):
              if cut_index < len(cut_val):
                  #direction (int)  => 型抜き後移動する方向(0:上 1:下 2:左 3:右)
                  if direction == 0:
                      if 0 <= x_boardsize - X_SIZE + i < x_boardsize and 0 <= Y + j < y_boardsize and board[x_boardsize - X_SIZE + i][Y + j] == -1:
                          board[x_boardsize - X_SIZE + i][Y + j] = cut_val[cut_index]
                          cut_index += 1

                  elif direction == 1:
                      if 0 <= i < x_boardsize and 0 <= Y + j < y_boardsize and board[i][Y + j] == -1:
                          board[i][Y + j] = cut_val[cut_index]
                          cut_index += 1

                  elif direction == 2:
                      if 0 <= X + i < x_boardsize and 0 <= y_boardsize - Y_SIZE + j < y_boardsize and board[X + i][y_boardsize - Y_SIZE + j] == -1:
                          board[X + i][y_boardsize - Y_SIZE + j] = cut_val[cut_index]
                          cut_index += 1

                  elif direction == 3:
                      if 0 <= X + i < x_boardsize and 0 <= j < y_boardsize and board[X + i][j] == -1:
                          board[X + i][j] = cut_val[cut_index]
                          cut_index += 1

      end = time()
      # print(f"action time : {end-start}")

      return board


    def reward(self, before_state, state, action=None):
        """
        現在の状態と入力された状態を基に報酬を計算
        一致数を増やすことを重視し、減少しても最終的に改善が見られればペナルティを軽減
        """
        H, W = len(state), len(state[0])
        total_pieces = H * W
        num_eq = 0  # 現在の一致数
        num_eq_before = 0  # 前回の状態との一致数

        # 一致数を計算
        for i in range(H):
            for j in range(W):
                if state[i][j] == self.goal[i][j]:
                    num_eq += 1
                if state[i][j] == before_state[i][j]:
                    num_eq_before += 1

        # 報酬計算
        progress = (num_eq - self.default_eq) / total_pieces
        this_rew = progress * 100  # 進捗に基づいた報酬

        # 一致数が減少してもペナルティは軽減する
        if num_eq < self.before_eq:
            this_rew -= 5  # 一時的な減少に対して軽いペナルティ

        # 全一致時の大きな報酬
        if num_eq == total_pieces:
            this_rew += 1000  # 全一致時のボーナス

        # 一致数が改善している場合は報酬を増加
        if num_eq > self.before_eq:
            this_rew += 50  # 改善が見られる場合のボーナス

        # 前回の一致数を更新
        self.num_eq = num_eq / total_pieces
        self.before_eq = num_eq

        # デバッグ用の表示
        if self.num_step % 100 == 0:
            print(f"{self.num_step} this_rew : {this_rew}")

        # テストモードの場合の最良の報酬とステップの更新
        if self.test and num_eq > self.max_eq:
            self.max_rew = this_rew
            self.max_eq = num_eq
            self.best_step = self.num_step
            self.ans_board = copy.deepcopy(state)

        return this_rew

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        # self.board = copy.deepcopy(self.start)
        if self.ans_board is not None:
            self.board = copy.deepcopy(self.ans_board) #訓練用
        else:
            self.board = copy.deepcopy(self.start)

        self.default_eq = 0
        for i in range(self.x_boardsize):
          for j in range(self.y_boardsize):
            if self.board[i][j] == self.goal[i][j]:
              self.default_eq += 1

        self.before_eq = self.default_eq
        
        self.num_step = 0
        self.done = False

        self.rew = 0
        self.before_rew = 0

        self.max_rew = -10000
        self.max_eq  = -1
        self.num_eq = 0
        self.best_step = 0
        self.before_act = None
        self.ans_board = None

        self.ans = {"n":0, "ops":[]}

        return self.board

    def action_sample(self):
        """
        ランダムにactionを出力する
        出力するactionの形式は、[256, 256, num_of_cutter, 4]
        """

        action = []

        action.append(random.randint(0,self.x_boardsize-1))
        action.append(random.randint(0,self.y_boardsize-1))
        action.append(random.randint(0,self.num_of_cutter-1))
        action.append(random.randint(0,3))

        return action

"""## 学習用class定義"""

"""松尾研講義用コードを基本実装に使用"""

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**3, num_eval_episodes=3):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': [], 'best_step': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval

        self.max_eq = 0

        self.befor_state = None

    def train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        self.evaluate(0)

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, self.befor_state, state, t, steps)
            self.befor_state = copy.deepcopy(state)
            if steps % 1000 == 0:
                print(steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                for _ in range(2):
                 self.algo.update(self.env)

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0 and steps >= self.algo.start_steps:
                self.evaluate(steps)

                if self.max_eq < self.env_test.max_eq:
                  self.max_eq = self.env_test.max_eq
                  model_path = path + f'model_{self.env_test.max_eq},{self.env_test.state_shape[0] * self.env_test.state_shape[1]}'
                  torch.save(self.algo.actor.state_dict(), model_path + '_actor')
                  torch.save(self.algo.critic.state_dict(), model_path + '_critic')
                  torch.save(self.algo.critic_target.state_dict(), model_path + '_critic_target')


    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        state = self.env_test.reset()
        befor = None
        done = False
        total_reward = 0

        evaluate_step = 0

        num_eq_before = 0
        rand = False
        while (not done):
            # print("evaluate step ", evaluate_step)
            evaluate_step += 1
            action = self.algo.exploit(befor, state, self.env_test.goal, rand=rand)
            befor = state
            state, reward, done, _ = self.env_test.step(action)
            total_reward += reward

            if rand:
                rand=False
            
            if abs(num_eq_before - self.env_test.num_eq) < 0.0005:
                rand = True

            num_eq_before = self.env_test.num_eq

        print(f"total reward   is {total_reward}")
        print(f"max reward     is {self.env_test.max_rew}")
        print(f"max eq         is {self.env_test.max_eq}")
        print(f"best step      is {self.env_test.best_step}")
        print(f"evalueate time is {self.time}")

        self.returns['step'].append(steps)
        self.returns['return'].append(self.env_test.max_rew)
        self.returns['best_step'].append(self.env_test.best_step)

        self.env_test.ans["n"] = self.env_test.best_step

        output = {"n":self.env_test.best_step, "ops": []}

        for i in range(self.env_test.best_step):
          output["ops"].append(self.env_test.ans["ops"][i])


        # 0~3の値に対応する色を定義
        cmap = ListedColormap(['red', 'green', 'blue', 'yellow'])

        # 図を描画
        plt.imshow(self.env_test.ans_board, cmap=cmap, interpolation='none')

        # カラーバーを表示して、各色が何の値に対応するかを表示
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['0', '1', '2', '3'])  # ラベルを設定

        # グリッド線を追加
        plt.grid(False)  # グリッドを非表示にする場合はTrueをFalseに変更

        # 図を表示
        plt.show()

        json_name = f"./solution_{self.env_test.max_eq},{self.env_test.state_shape[0] * self.env_test.state_shape[1]}_{self.env_test.best_step}.json"

        with open(json_name, 'w') as f:
            json.dump(output, f, indent=2)

    """
        # solution.json を読み込む
        with open(json_name, 'r') as f:
  	        solution_data = json.load(f)

	    # ヘッダーの設定
        headers = {"Content-Type": "application/json", "Procon-Token": "osaka534508e81dbe6f70f9b2e07e61464780bd75646fdabf7b1e7d828d490e3"}

	    # POST リクエストを送信
        response = requests.post(host_name + "/answer", headers=headers, json=solution_data)

	    # レスポンスのステータスコードと内容を確認
        print("Status Code:", response.status_code)
	"""

    def plot_return(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Max Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'Best-Reward about action', fontsize=24)
        plt.tight_layout()

    def plot_steps(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['best_step'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Best Step', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'Least step about best reward', fontsize=24)
        plt.tight_layout()

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))
"""## 学習用アルゴリズム定義"""

class Algorithm(ABC):

    def explore(self, befor, state, goal, batch_size=1):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． 
            本メソッドに関しても基本はexploitのように決定的にしてみてもよいかもしれない
        """
        if befor[0] is None:
            print(state[0].shape)
            befor[0] = torch.zeros_like(torch.tensor(state[0]))

        state = torch.tensor(state, dtype=torch.float, device=self.device).clone().detach().unsqueeze_(0)
        goal  = torch.tensor(goal, dtype=torch.float, device=self.device).clone().detach().unsqueeze_(0)
        befor = torch.tensor(befor, dtype=torch.float, device=self.device).clone().detach().unsqueeze_(0)
        # with torch.no_grad():
        action_pos, log_pi_pos, action_sellect, log_pi_sellect, action_direct, log_pi_direct = self.actor.sample(befor, state, goal, batch_size)


        action_pos, action_sellect, action_direct = action_pos + abs(torch.min(action_pos.view(-1))) + 1e-3, action_sellect + abs(torch.min(action_sellect.view(-1)))+1e-3, action_direct + abs(torch.min(action_direct.view(-1)))+1e-3

        action_pos_list = []
        action_pos_x = []
        action_pos_y = []
        for i in range(batch_size):
            action_pos_list.append(random.choices(self.pos_list, k=1, weights=mySqueeze(action_pos[i].view(-1).detach().cpu().numpy().tolist()))[0])
            action_pos_x.append(action_pos_list[i] // len(state[0]))
            action_pos_y.append(action_pos_list[i] % len(state[0][0]))



        action_sellect_list = []
        action_direct_list = []
        if batch_size != 1:

          for i in range(batch_size):
            action_sellect_list.append(random.choices(self.sellect_list, k=1, weights=mySqueeze(action_sellect[i].cpu().detach().numpy().tolist()))[0])
            action_direct_list.append(random.choices(self.direct_list, k=1, weights=mySqueeze(action_direct[i].cpu().detach().numpy().tolist()))[0])

        else:

          action_sellect_list = random.choices(self.sellect_list, k=1, weights=mySqueeze(action_sellect.cpu().detach().numpy().tolist()))[0]
          action_direct_list = random.choices(self.direct_list, k=1, weights=mySqueeze(action_direct.cpu().detach().numpy().tolist()))[0]


        if batch_size != 1:
            action = [[action_pos_x[i], action_pos_y[i], action_sellect_list[i], action_direct_list[i]] for i in range(batch_size)]
            log_pi = torch.cat([log_pi_pos, log_pi_sellect, log_pi_direct], dim=0).view(-1)
        else:
            action = [int(mySqueeze(action_pos_x)), int(mySqueeze(action_pos_y)), int(mySqueeze(action_sellect_list)), int(mySqueeze(action_direct_list))]
            log_pi = torch.cat([log_pi_pos, log_pi_sellect, log_pi_direct], dim=0).view(-1)

        return action, log_pi

    def exploit(self, befor, state, goal, rand = False):
        if befor is None:
            befor = torch.zeros_like(torch.tensor(state))

        # print(f"state shape = ({len(state)}, {len(state[0])})")
        # print(f"goal shape = ({len(goal)}, {len(goal[0])})")
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        goal  = torch.tensor(goal, dtype=torch.float, device=self.device).unsqueeze_(0)
        befor = torch.tensor(befor, dtype=torch.float, device=self.device).unsqueeze_(0)

        if not random:
            with torch.no_grad():
                action_pos, action_sellect, action_direct = self.actor(befor, state, goal)
                action_pos, action_sellect, action_direct = action_pos + abs(torch.min(action_pos.view(-1))) + 1e-3, action_sellect + abs(torch.min(action_sellect.view(-1)))+1e-3, action_direct + abs(torch.min(action_direct.view(-1)))+1e-3
                action_pos = torch.argmax(action_pos.view(1,-1), dim=-1).cpu().numpy()
                # print(f"action_pos = {action_pos}")


                action_pos_x = action_pos // len(state[0])
                action_pos_y = action_pos % len(state[0][0])

                print(f"act_x = {action_pos_x}, act_y = {action_pos_y}")

                action_sellect = torch.argmax(action_sellect, dim=-1).cpu().numpy()

                action_direct = torch.argmax(action_direct, dim=-1).cpu().numpy()


                action = [int(action_pos_x), int(action_pos_y), int(action_sellect), int(action_direct)]
        else:
            with torch.no_grad():
                action_pos, action_sellect, action_direct = self.actor(befor, state, goal)
                action_pos, action_sellect, action_direct = action_pos + abs(torch.min(action_pos.view(-1))) + 1e-3, action_sellect + abs(torch.min(action_sellect.view(-1)))+1e-3, action_direct + abs(torch.min(action_direct.view(-1)))+1e-3

                action_pos = random.choices(self.pos_list, k=1, weights=action_pos.view(-1).detach().cpu().numpy().tolist())[0]

                # print(f"action_pos = {action_pos}")


                action_pos_x = action_pos // len(state[0])
                action_pos_y = action_pos % len(state[0][0])

                print(f"act_x = {action_pos_x}, act_y = {action_pos_y}")

                action_sellect = random.choices(self.sellect_list, k=1, weights=action_sellect.view(-1).cpu().detach().numpy().tolist())[0]

                action_direct = random.choices(self.direct_list, k=1, weights=action_direct.view(-1).cpu().detach().numpy().tolist())[0]


                action = [int(action_pos_x), int(action_pos_y), int(action_sellect), int(action_direct)]


        return action

    @abstractmethod
    def is_update(self, steps):
        """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
        """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass

"""**SACの実装**

### network構築1

### networkの構築2

qiita 『【Pytorch】UNetを実装する』（https://qiita.com/gensal/items/03e9a6d0f7081e77ba37）　参照
"""

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.LeakyReLU()
        # self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class SACActor(nn.Module):

    """
    抜き型のサイズごとに
    """

    def __init__(self, num_of_cutter):
        super().__init__()

        self.num_of_cutter = num_of_cutter

        """
        Unetを使用してネットワークを一つだけに絞る
        """

        self.TCB1 = TwoConvBlock(3, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        # self.TCB5 = TwoConvBlock(128, 256, 256)

        self.linear_sellect = nn.Linear(128, 2 * 100)
        self.linear_direction = nn.Linear(128, 2 * 4)

        # self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.UC1 = UpConv(256, 128)
        self.UC2 = UpConv(128, 64)
        self.UC3 = UpConv(64, 32)
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 2, kernel_size = 1)

    def forward(self, befor, state, goal, batch_size=1):
        #各配列を結合しやすいように形成する
        
        if befor.dim() > 3:
          befor = befor.squeeze()

        if state.dim() > 3:
          state = state.squeeze()

        if goal.dim() > 3:
          goal = goal.squeeze()

        befor, state, goal = befor.unsqueeze(1), state.unsqueeze(1), goal.unsqueeze(1)

        state = torch.cat([befor, state, goal], dim=1) #データを結合

        #データをネットワークに通す
        state = self.TCB1(state)
        x1 = state
        state = self.maxpool(state)

        state = self.TCB2(state)
        x2 = state
        state = self.maxpool(state)

        state = self.TCB3(state)
        x3 = state
        state = self.maxpool(state)

        state = self.TCB4(state)

        x_sellect_direction = self.gap(state).view(batch_size, -1)
        return_sellect = self.linear_sellect(x_sellect_direction).chunk(2, dim=-1)[0]
        return_direct = self.linear_direction(x_sellect_direction).chunk(2, dim=-1)[0]

        return_sellect = torch.tanh(return_sellect)[0][0:self.num_of_cutter].unsqueeze_(0)

        return_direct = torch.tanh(return_direct)

        state = self.UC2(state)
        state = torch.cat([x3, state], dim = 1)
        state = self.TCB7(state)

        state = self.UC3(state)
        state = torch.cat([x2, state], dim = 1)
        state = self.TCB8(state)

        state = self.UC4(state)
        state = torch.cat([x1, state], dim = 1)
        state = self.TCB9(state)

        state = self.conv1(state)

        return_pos = torch.tanh(state.chunk(2, dim=1)[0].view(batch_size, -1))

        return return_pos, return_sellect, return_direct

    def sample(self, befor, state, goal, batch_size=1):

        if befor.dim() > 3:
          befor = befor.squeeze() 

        if state.dim() > 3:
          state = state.squeeze()

        if goal.dim() > 3:
          goal = goal.squeeze()

        befor, state, goal = befor.unsqueeze(1), state.unsqueeze(1), goal.unsqueeze(1)

        state = torch.cat([befor, state, goal], dim=1)

        state = self.TCB1(state)
        x1 = state
        state = self.maxpool(state)

        state = self.TCB2(state)
        x2 = state
        state = self.maxpool(state)

        state = self.TCB3(state)
        x3 = state
        state = self.maxpool(state)

        state = self.TCB4(state)

        x_sellect_direction = self.gap(state).view(batch_size, -1)
        means_sellect, log_stds_sellect = self.linear_sellect(x_sellect_direction).chunk(2, dim=-1)
        means_sellect, log_stds_sellect = means_sellect[:, 0:self.num_of_cutter], log_stds_sellect[:, 0:self.num_of_cutter]
        means_direct, log_stds_direct = self.linear_direction(x_sellect_direction).chunk(2, dim=-1)


        state = self.UC2(state)

        state = torch.cat([x3, state], dim = 1)
        state = self.TCB7(state)

        state = self.UC3(state)
        state = torch.cat([x2, state], dim = 1)
        state = self.TCB8(state)

        state = self.UC4(state)
        state = torch.cat([x1, state], dim = 1)
        state = self.TCB9(state)

        state = self.conv1(state)

        means_pos, log_stds_pos = state.view(batch_size, -1).chunk(2, dim=-1)


        action_pos, log_pi_pos = reparameterize(means_pos, log_stds_pos.clamp(-20, 2))
        action_sellect, log_pi_sellect = reparameterize(means_sellect, log_stds_sellect.clamp(-20, 2))
        action_direct, log_pi_direct = reparameterize(means_direct, log_stds_direct.clamp(-20, 2))

        return  action_pos, log_pi_pos, action_sellect, log_pi_sellect, action_direct, log_pi_direct#actions, log_pis を出力

class SACCritic(nn.Module):

    def __init__(self):
        super().__init__()

        self.TCB1 = TwoConvBlock(3, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(128, 2)

    def forward(self, state, after_action, goal, batch_size=1):

        if state.dim() > 3:
          state = state.squeeze()

        if after_action.dim() > 3:
          after_action = after_action.squeeze()

        if goal.dim() > 3:
          goal = goal.squeeze()

        state, after_action, goal = state.unsqueeze(1), after_action.unsqueeze(1), goal.unsqueeze(1) # mini-batchに対応させる

        state = torch.cat([state, after_action, goal], dim=1)

        #データをネットワークに通す
        state = self.TCB1(state)
        state = self.maxpool(state)

        state = self.TCB2(state)
        state = self.maxpool(state)

        state = self.TCB3(state)
        state = self.maxpool(state)

        state = self.TCB4(state)

        state = self.gap(state).view(batch_size, -1)

        state = self.linear(state)

        return state

"""### アルゴリズムと記憶領域の構築"""

class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, batch_size, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        self.batch_size = batch_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.goal = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, goal):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.goal[self._p].copy_(torch.from_numpy(goal))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        # idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes = np.arange(batch_size) + np.random.randint(low=0, high=self._n // batch_size, size=1) * self.batch_size
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
            self.goal[idxes]
        )

class SAC(Algorithm):

    def __init__(self, state_shape, action_shape, num_of_cutter, device=torch.device('cuda'), seed=0,
                 batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 replay_size=10**4, start_steps=2*10**2, tau=5e-3, alpha=0.2, reward_scale=1.0,
                 pretrain = False, model_weight_name = None):
        super().__init__()

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            batch_size = batch_size,
            device=device,
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = SACActor(
            num_of_cutter=num_of_cutter
        ).to(device)


        self.critic = SACCritic().to(device)

        self.critic_target = SACCritic().to(device).eval()

        if pretrain:

          model_path = path + model_weight_name


          self.actor.load_state_dict(torch.load(model_path + '_actor'))
          self.critic.load_state_dict(torch.load(model_path + '_critic'))
          self.critic_target.load_state_dict(torch.load(model_path + '_critic_target'))

        else:
          # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
          self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False


        # オプティマイザ．
        self.optim_actor = torch.optim.RMSprop(self.actor.parameters(), lr=lr_actor)

        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.actor_update_count = 0

        self.pos_list     = list(range(state_shape[0]*state_shape[1]))
        self.sellect_list = list(range(num_of_cutter))
        self.direct_list  = list(range(4))

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= self.batch_size and steps % self.batch_size == 0

    def step(self, env, befor, state, t, steps):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する
        if steps <= self.start_steps:
            action = env.action_sample()
        else:
            action, _ = self.explore(befor, state, env.goal)
        next_state, reward, done, _ = env.step(action)


        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        send_state, send_action, send_next_state, send_goal = np.array(state).astype(float), np.array(action).astype(float), np.array(next_state).astype(float), np.array(env.goal).astype(float)

        # リプレイバッファにデータを追加する．
        self.buffer.append(send_state, send_action, reward, done_masked, send_next_state, send_goal)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, env):
        print("start update")

        self.learning_steps += 1
        states, _, rewards, dones, next_states, goal = self.buffer.sample(self.batch_size)

        self.update_critic(states, rewards, dones, next_states, goal, env)

        if self.actor_update_count == 1:
            self.update_actor(states, goal, env)
            self.actor_update_count = 0
        else:
            self.actor_update_count += 1

        self.update_target()

        print("end update")

    def update_critic(self, states, rewards, dones, next_states, goal, env):
        curr_qs = self.critic(states, next_states, goal, batch_size = self.batch_size)

        # with torch.no_grad():
        next_actions, log_pis = self.explore(states, next_states, goal, self.batch_size)
        curr_qs1, curr_qs2 = curr_qs.chunk(2, dim=-1)
        curr_qs1 = torch.clamp(curr_qs1, min=-10**9, max=10**9)
        curr_qs2 = torch.clamp(curr_qs2, min=-10**9, max=10**9) 

        """
        env.actionはいじらずにここでbatch対応させる。
        networkについてはbatch対応させる（これに関しては計算効率向上）
        """
        #tensorで初期化しないとエラーの原因になる
        # after_next_actions = torch.empty((self.batch_size, next_states[0].shape[0], next_states[0].shape[1]), dtype=torch.float, device=self.device)

        act_start = time()

        after_next_actions_last = env.action(next_states[self.batch_size - 1], [next_actions[self.batch_size-1][0], next_actions[self.batch_size-1][1]], env.cutter[next_actions[self.batch_size-1][2]], next_actions[self.batch_size-1][3])
        after_next_actions      = torch.cat([next_states[1:self.batch_size], after_next_actions_last.unsqueeze_(0)], dim=0)

        act_end = time()
        print(f"act time is {act_end - act_start}")


        next_qs = self.critic_target(next_states, after_next_actions, goal, batch_size = self.batch_size)

        next_qs = torch.tensor(torch.min(next_qs, dim=0).values.mean()) - self.alpha * torch.tensor(log_pis).mean()


        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        target_qs = torch.clamp(target_qs, min=-10**9, max=10**9) 

        loss_critic1 = (curr_qs1.mean() - target_qs).pow_(2).mean() / 10**5
        loss_critic2 = (curr_qs2.mean() - target_qs).pow_(2).mean() / 10**5

        print(f"critic loss = {(loss_critic1 + loss_critic2).mean()}")

        critic_update_start = time()

        self.optim_critic.zero_grad()


        (loss_critic1 + loss_critic2).mean().backward(retain_graph=True)


        # for name, param in self.critic.named_parameters():
        #   if param.grad is not None:
        #       print(f"Gradients for {name}: {param.grad.mean()}")
        #   else:
        #       print("grad is None")

        self.optim_critic.step()

        critic_update_end = time()
        print(f"critic backward time is {critic_update_end - critic_update_start}")


    def update_actor(self, states, goal, env):
        befor_states = torch.zeros_like(states)
        for i in range(len(states)-1):
            befor_states[i+1] = states[i]
        actions, log_pis = self.explore(befor_states, states, goal, self.batch_size)
        # next_states = torch.empty((self.batch_size, states[0].shape[0], states[0].shape[1]), dtype=torch.float, device=self.device)

        act_start = time()

        after_actions_last = env.action(states[self.batch_size - 1], [actions[self.batch_size-1][0], actions[self.batch_size-1][1]], env.cutter[actions[self.batch_size-1][2]], actions[self.batch_size-1][3])
        next_states      = torch.cat([states[1:self.batch_size], after_actions_last.unsqueeze_(0)], dim=0)

        # for i in range(self.batch_size):
        #     if i == self.batch_size - 1:
        #       next_states[i] = env.action(states[i], [actions[i][0], actions[i][1]], env.cutter[actions[i][2]], actions[i][3])
        #       break
        #     next_states[i] = states[i+1]

        act_end = time()
        print(f"act time is {act_end - act_start}")

        qs = self.critic(states, next_states, goal, batch_size = self.batch_size)
        loss_actor = (self.alpha * log_pis.mean() - torch.min(qs, dim=-1).values.mean()).mean()
        print(f"actor loss = {loss_actor}")

        actor_update_start = time()
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=True)

        # for name, param in self.actor.named_parameters():
        #   if param.grad is not None:
        #       print(f"Gradients for {name}: {param.grad.mean()}")

        #   else:
        #       print("gradient is None")

        self.optim_actor.step()
        actor_update_end = time()
        print(f"actor backward time is {actor_update_end - actor_update_start}")

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)     #パラメータtを引数倍する
            t.data.add_(self.tau * s.data)  #パラメータtに引数分の加減をおこなう
