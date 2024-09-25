from abc import ABC, abstractmethod
from time import time
from datetime import timedelta
import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gc
from pprint import pprint
import json

import torch
import torchvision
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from torchvision import models


def mySqueeze(x):
  return np.array(x).squeeze().tolist()

def resize(board, batch_size, shape=[256,256]):

    #評価時にだけboardが三次元(1,32,32)の形で渡されているのでsqueeze等の対策が必要

    start = time()

    if not isinstance(board, list):
      board = board.tolist()

    board_copy = copy.deepcopy(board)
    board_copy = mySqueeze(board_copy)

    if np.array(board_copy).ndim <=2:
        board_copy = np.expand_dims(np.array(board_copy), 0).tolist()

    init_tensor = torch.full([batch_size, shape[0], shape[1]],fill_value=-1, dtype=torch.float)

    for batch in range(batch_size):
      for i in range(len(board)):
        for j in range(len(board[0])):
          init_tensor[batch][i][j] = float(board_copy[batch][i][j])

    end = time()

    # print(f"risize time : {end-start}")
    return init_tensor

#2重ループが頻出しており、実行時間が長いと考えられるので要改善
class transition():
    """
    本クラスはenv(学習環境)として扱うクラスであるので、
    行動に対して次の状態と報酬、実行が終わったかどうかを返す。
    """
    def __init__(self, board, cutter, goal, frequ=1, test=False):
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
        self.goal  = copy.deepcopy(goal)
        self.cutter = copy.deepcopy(cutter)
        self.done  = False
        self.num_step = 0
        self._max_episode_steps = 1000
        self.frequ = frequ #frequは報酬の獲得頻度を表す。
        self.before_rew = 0

        self.rew   = 0
        self.h_countforrew, self.w_countforrew = 0, 0 # 左or上詰めで、行・列の一致に報酬を与えるのでそのindexを定義しておく

        self.num_of_cutter = len(self.cutter)
        self.state_shape  = [len(board), len(board[0])]
        self.action_shape = [4]

        self.max_rew = -10000
        self.max_eq  = -1
        self.best_step = 0

        self.ans = {"n":0, "ops":[]}

        self.ans_board = None

        self.test = test

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
            self.rew += self.reward(state,next_state)
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

        #next_state : 次の状態, self.rew : 今回の行動への報酬, self.done : 行動が終了するかどうか
        return next_state, self.rew, self.done, self.num_step

    def action(self, board ,pos, cutter, direction, batch_size=1):
        """
        本メソッドでは、選択した行動の結果から次の状態を計算する

        cutter(2次元list) => 抜き型の状態(0,1の二値 1の箇所を抜く)
        pos   (1次元list) => 型抜きを行う座標[x,y]
        direction (int)  => 型抜き後移動する方向(0:上 1:下 2:左 3:右)
        """
        start = time()
        #cutterのサイズと型抜き座標を定数として格納
        # X_SIZE, Y_SIZE = cutter.size(), cutter[0].size()
        X_SIZE, Y_SIZE = len(cutter), len(cutter[0])
        X, Y = pos[0], pos[1]
        self.x_boardsize, self.y_boardsize = len(self.board), len(self.board[0])

        # X = (X * self.x_boardsize) // 256
        # Y = (Y * self.y_boardsize) // 256

        #どの方向に動かすかをbool値で格納
        up, down, left, right = direction==0, direction==1, direction==2, direction==3

        #cut_val には抜いたピースを一時的に格納(早く抜いたピースほどindexが小さい)
        #包括表記を使用しなければ全行(cut_val[0] ~ cut_val[255])が全て同じ値となってしまうので注意
        cut_val = [[-1] * 256 for _ in range(256)]
        x_count, y_count = 0,0
        #抜くピースを格納し、穴あきとなる箇所の値を-1で埋める
        for i in range(X_SIZE):
            x_bool = False
            for j in range(Y_SIZE):
                #現在のマスがgirdの範囲内かどうか確認
                if not(0<= X+i <self.x_boardsize and 0<= Y+j <self.y_boardsize): continue

                #抜き型のうち1のマスに当たる箇所を抜く　値は保存
                if cutter[i][j] == 1:
                    if up or down:
                        cut_val[x_count][y_count] = board[X+i][Y+j]
                    elif left or right:
                        cut_val[x_count][y_count] = board[X+i][Y+j]

                    board[X+i][Y+j] = -1

                    y_count += 1
                    x_bool = True


            if x_bool:
                x_count += 1

            y_count = 0


        #countには-1が続いた回数(必要移動回数)を格納するが、必要サイズが変動するので最大の256を取っておく
        count = [0] * 256
        for i in range(self.x_boardsize):
            for j in range(self.y_boardsize):
                #以下はそれぞれ、数値を確認していく方向が共通な上左と下右それぞれについての現在のマスがgridの範囲内か否かを判断するbool値

                """
                以下の条件分岐は次のような分類を行っている

                どの方向に移動するか：
                    現在のマスは範囲内か：範囲内でなければcontinue

                    現在のマスは -1 か：-1 なら count+=1 で continue
                    (-1以外なら)現在のマスの数字をcount分前のマスに挿入し、現在のマスは -1 に
                """

                if up:
                    x_index, y_index = i, j

                    if board[x_index][y_index] == -1:
                        count[y_index] += 1

                    elif count[y_index] != 0:
                        board[x_index - count[y_index]][y_index] = board[x_index][y_index]
                        board[x_index][y_index] = -1

                elif left:
                    x_index, y_index = i, j

                    if board[x_index][y_index] == -1:
                        count[x_index] += 1

                    elif count[x_index] != 0:
                        board[x_index][y_index-count[x_index]] = board[x_index][y_index]
                        board[x_index][y_index] = -1


                elif down:
                    x_index, y_index = self.x_boardsize-1-i, self.y_boardsize-1-j

                    if board[x_index][y_index] == -1:
                        count[y_index] += 1
                    elif count[y_index] != 0:
                        board[x_index +count[y_index]][y_index] = board[x_index][y_index]
                        board[x_index][y_index] = -1

                elif right:
                    x_index, y_index = self.x_boardsize-1-i, self.y_boardsize-1-j

                    if board[x_index][y_index] == -1:
                        count[x_index] += 1
                    elif count[x_index] != 0:
                        board[x_index][y_index +count[x_index]] = board[x_index][y_index]
                        board[x_index][y_index] = -1

        x_count, y_count = 0,0
        #以下ではgridに対して全探索を行っているがおいおい適切な範囲を考える必要あり
        for i in range(self.x_boardsize):
            x_bool = False
            for j in range(self.y_boardsize):
                if board[i][j] != -1:
                    continue
                elif board[i][j] == -1:
                    board[i][j] = cut_val[x_count][y_count]

                    # print(x_count,y_count,cut_val[x_count][y_count])
                    x_bool = True
                    y_count += 1

            if x_bool:
                x_count += 1
            y_count = 0

            end = time()

            # print(f"action time : {end-start}")

        return board

    def reward(self, befor_state, state):
        """
        本メソッドでは、現在の状態(board)と入力された状態(state)をもとに報酬を計算する
        現段階では、一度に渡す報酬の最大値を1000最小値(ターンごとのペナルティ)を-1とする。
        """
        H, W = len(state), len(state[0])
        this_rew = 0 #本stepで獲得した報酬を格納
        num_eq = 0
        num_eq_befor = 0

        for i in range(H):
            for j in range(W):
                if state[i][j] == self.goal[i][j]:
                    num_eq += 1

                if state[i][j] == befor_state[i][j]:
                    num_eq_befor += 1

        if num_eq == H*W:
            this_rew += 1000

        else:
                this_rew += num_eq / (H*W) * 1000

                # if num_eq_befor == H*W:
                #   this_rew = -1000



        keep_rew = this_rew

        # if this_rew <= self.before_rew:
        #     this_rew = -1 #前回以下の報酬しか得れていないのなら負の報酬を与える。ー＞より大きい報酬を得られるように学習

        self.before_rew = keep_rew

        if self.test and this_rew >= self.max_rew:
          self.max_rew = this_rew

          self.max_eq = num_eq

          self.best_step = self.num_step

          self.ans_board = copy.deepcopy(state)


        return this_rew

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        self.board = copy.deepcopy(self.start)
        self.num_step = 0
        self.done = False

        self.rew = 0
        self.h_countforrew, self.w_countforrew = 0, 0

        self.max_rew = -10000
        self.max_eq  = -1
        self.best_step = 0

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

    def train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, state, t, steps)
            print(steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update(self.env)

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0 and steps >= self.algo.start_steps:
                self.evaluate(steps)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        state = self.env_test.reset()
        done = False

        evaluate_step = 0
        while (not done):
            print("evaluate step ", evaluate_step)
            evaluate_step += 1
            action = self.algo.exploit(state, self.env_test.goal)
            state, reward, done, _ = self.env_test.step(action)

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

        with open("./test_initial.json", 'w') as f:
            json.dump(output, f, indent=2)

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

class Algorithm(ABC):

    def explore(self, state, goal, batch_size=1):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action_pos, log_pi_pos, action_sellect, log_pi_sellect, action_direct, log_pi_direct = self.actor.sample(state, goal, batch_size)

            action_pos, action_sellect, action_direct = torch.exp(action_pos), torch.exp(action_sellect), torch.exp(action_direct)

            action_pos_list = []
            action_pos_x = []
            action_pos_y = []

            if batch_size != 1:
              for i in range(batch_size):
                action_pos_list.append(random.choices(self.pos_list, k=1, weights=mySqueeze(action_pos[i].cpu().numpy().tolist()))[0])
                action_pos_x.append(action_pos_list[i] % len(state[0][0]))
                action_pos_y.append(action_pos_list[i] // len(state[0]))
            else:
                action_pos_list = random.choices(self.pos_list, k=1, weights=mySqueeze(action_pos.cpu().numpy().tolist()))[0]
                action_pos_x = action_pos_list % len(state[0][0])
                action_pos_y = action_pos_list // len(state[0])



            action_sellect_list = []
            action_direct_list = []
            if batch_size != 1:

              for i in range(batch_size):
                action_sellect_list.append(random.choices(self.sellect_list, k=1, weights=mySqueeze(action_sellect[i].cpu().numpy().tolist()))[0])
                action_direct_list.append(random.choices(self.direct_list, k=1, weights=mySqueeze(action_direct[i].cpu().numpy().tolist()))[0])

            else:

              action_sellect_list = random.choices(self.sellect_list, k=1, weights=mySqueeze(action_sellect.cpu().numpy().tolist()))[0]
              action_direct_list = random.choices(self.direct_list, k=1, weights=mySqueeze(action_direct.cpu().numpy().tolist()))[0]

            if batch_size != 1:
               action = [[action_pos_x[i], action_pos_y[i], action_sellect_list[i], action_direct_list[i]] for i in range(batch_size)]
               log_pi = torch.tensor([[*log_pi_pos[i].cpu().numpy().tolist(), *log_pi_sellect[i].cpu().numpy().tolist(), *log_pi_direct[i].cpu().numpy().tolist()] for i in range(batch_size)])
            else:
               action = [int(action_pos_x), int(action_pos_y), int(action_sellect_list), int(action_direct_list)]
               log_pi = torch.tensor([*log_pi_pos.cpu().numpy().tolist(), *log_pi_sellect.cpu().numpy().tolist(), *log_pi_direct.cpu().numpy().tolist()])

        return action, mySqueeze(log_pi)

    def exploit(self, state, goal):
        """ 決定論的な行動を返す． """
        # print(f"state shape = ({len(state)}, {len(state[0])})")
        # print(f"goal shape = ({len(goal)}, {len(goal[0])})")
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action_pos, action_sellect, action_direct = self.actor(state, goal)
            action_pos = torch.argmax(action_pos, dim=-1).cpu().numpy()

            # print(f"state shape = ({len(state)}, {len(state[0])})")

            # print(f"action_pos = {action_pos}")


            action_pos_x = action_pos % len(state[0][0])
            action_pos_y = action_pos // len(state[0])

            print(f"act_x = {action_pos_x}, act_y = {action_pos_y}")

            action_sellect = torch.argmax(action_sellect, dim=-1).cpu().numpy()

            action_direct = torch.argmax(action_direct, dim=-1).cpu().numpy()


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
        self.rl = nn.ReLU()
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

        """
        Unetを使用してネットワークを一つだけに絞る
        """

        self.TCB1 = TwoConvBlock(2, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.TCB5 = TwoConvBlock(128, 256, 256)

        self.linear_sellect = nn.Linear(256, 2 * num_of_cutter)
        self.linear_direction = nn.Linear(256, 2 * 4)

        self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.UC1 = UpConv(256, 128)
        self.UC2 = UpConv(128, 64)
        self.UC3 = UpConv(64, 32)
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 2, kernel_size = 1)

    def forward(self, states, goal, batch_size=1):

        start = time()

        state     = torch.tensor(states).clone().to("cuda" if torch.cuda.is_available() else "cpu")
        goal_copy = torch.tensor(goal).clone().repeat(batch_size, 1, 1).to("cuda" if torch.cuda.is_available() else "cpu")

        #各配列を結合しやすいように形成する

        if state.dim() > 3:
          state = state.squeeze()

        if goal_copy.dim() > 3:
          goal_copy = goal_copy.squeeze()

        state, goal_copy = state.unsqueeze(1), goal_copy.unsqueeze(1)

        x = torch.cat([state, goal_copy], dim=1) #データを結合

        #データをネットワークに通す
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x_sellect_direction = self.gap(x).view(batch_size, -1)
        x_sellect = self.linear_sellect(x_sellect_direction).chunk(2, dim=-1)[0]
        x_direction = self.linear_direction(x_sellect_direction).chunk(2, dim=-1)[0]

        return_sellect = torch.tanh(x_sellect)
        return_direct = torch.tanh(x_direction)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return_pos = torch.tanh(x.chunk(2, dim=1)[0].view(batch_size, -1))


        end = time()

        # print(f"action forward time : {end-start}")
        return return_pos, return_sellect, return_direct

    def sample(self, states, goal, batch_size=1):

        state     = torch.tensor(states).clone().to("cuda" if torch.cuda.is_available() else "cpu")
        goal_copy = torch.tensor(goal).clone().to("cuda" if torch.cuda.is_available() else "cpu")

        if batch_size == 1:
          goal_copy = goal_copy.unsqueeze(0)

        if state.dim() > 3:
          state = state.squeeze()

        if goal_copy.dim() > 3:
          goal_copy = goal_copy.squeeze()

        state, goal_copy = state.unsqueeze(1), goal_copy.unsqueeze(1)

        x = torch.cat([state, goal_copy], dim=1)

        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x_sellect_direction = self.gap(x).view(batch_size, -1)
        means_sellect, log_stds_sellect = self.linear_sellect(x_sellect_direction).chunk(2, dim=-1)
        means_direct, log_stds_direct = self.linear_direction(x_sellect_direction).chunk(2, dim=-1)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        means_pos, log_stds_pos = x.view(batch_size, -1).chunk(2, dim=-1)


        action_pos, log_pi_pos = reparameterize(means_pos, log_stds_pos.clamp(-20, 2))
        action_sellect, log_pi_sellect = reparameterize(means_sellect, log_stds_sellect.clamp(-20, 2))
        action_direct, log_pi_direct = reparameterize(means_direct, log_stds_direct.clamp(-20, 2))

        return  action_pos, log_pi_pos, action_sellect, log_pi_sellect, action_direct, log_pi_direct#actions, log_pis を出力

class SACCritic(nn.Module):

    def __init__(self, side_size=256):
        super().__init__()

        #学習済みモデル使用用に書き換え

        # Resnet50を重み付きで読み込む
        self.net = models.resnet50(pretrained = True)

        # 最終ノードの出力を変更する
        self.net.fc = nn.Linear(self.net.fc.in_features,  2)

    def forward(self, states, after_actions, goal, batch_size=1):

        start = time()

        state        = torch.tensor(states).clone().to("cuda" if torch.cuda.is_available() else "cpu")
        after_action = torch.tensor(after_actions).clone().to("cuda" if torch.cuda.is_available() else "cpu")
        goal_copy    = torch.tensor(goal).clone().to("cuda" if torch.cuda.is_available() else "cpu")

        if state.dim() > 3:
          state = state.squeeze()

        if after_action.dim() > 3:
          after_action = after_action.squeeze()

        if goal_copy.dim() > 3:
          goal_copy = goal_copy.squeeze()

        state, after_action, goal_copy = state.unsqueeze(1), after_action.unsqueeze(1), goal_copy.unsqueeze(1) # mini-batchに対応させる

        x = torch.cat([state, after_action, goal_copy], dim=1)

        # state = resize(states, batch_size, shape=[256, 256]).to("cuda" if torch.cuda.is_available() else "cpu")
        # after_action = resize(after_actions, batch_size, shape=[256, 256]).to("cuda" if torch.cuda.is_available() else "cpu")
        # goal_copy = resize(goal, batch_size, shape=[256, 256]).to("cuda" if torch.cuda.is_available() else "cpu")

        # state, after_action, goal_copy = state.squeeze(), after_action.squeeze(), goal_copy.squeeze()
        # state, after_action, goal_copy = state.unsqueeze(1), after_action.unsqueeze(1), goal_copy.unsqueeze(1) # mini-batchに対応させる
        # x = torch.cat([state, after_action, goal_copy], dim=1)

        x = self.net(x)

        end = time()

        # print(f"critic forward time : {end-start}")
        return x

"""### アルゴリズムと記憶領域の構築"""

class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, goal, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

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
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
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
                 replay_size=10**3, start_steps=2*10**2, tau=5e-3, alpha=0.2, reward_scale=1.0):
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
            goal=state_shape,
            device=device,
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = SACActor(
            num_of_cutter=num_of_cutter
        ).to(device)


        self.critic = SACCritic().to(device)

        self.critic_target = SACCritic().to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False


        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

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

        self.pos_list     = list(range(state_shape[0]*state_shape[1]))
        self.sellect_list = list(range(num_of_cutter))
        self.direct_list  = list(range(4))

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する
        if steps <= self.start_steps:
            action = env.action_sample()
        else:
            action, _ = self.explore(state, env.goal)
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
        states, actions, rewards, dones, next_states, goal = self.buffer.sample(self.batch_size)

        self.update_critic(states, rewards, dones, next_states, goal, env)
        self.update_actor(states, goal, env)
        self.update_target()

        print("end update")

    def update_critic(self, states, rewards, dones, next_states, goal, env):
        curr_qs = self.critic(states, next_states, goal, batch_size = self.batch_size)

        with torch.no_grad():
            next_actions, log_pis = self.explore(next_states, goal, self.batch_size)
            curr_qs1, curr_qs2 = curr_qs.chunk(2, dim=-1)

            """
            env.actionはいじらずにここでbatch対応させる。
            networkについてはbatch対応させる（これに関しては計算効率向上）
            """
            #tensorで初期化しないとエラーの原因になる
            after_next_actions = torch.empty((self.batch_size, next_states[0].shape[0], next_states[0].shape[1]), dtype=torch.float, device=self.device)

            act_start = time()
            for i in range(self.batch_size):
              after_next_actions[i] = env.action(next_states[i], [next_actions[i][0], next_actions[i][1]], env.cutter[next_actions[i][2]], next_actions[i][3])

            act_end = time()
            print(f"act time is {act_end - act_start}")


            next_qs = self.critic_target(next_states, after_next_actions, goal, batch_size = self.batch_size)

            next_qs = torch.tensor(torch.min(next_qs, dim=-1).values.mean()) - self.alpha * torch.tensor(log_pis).mean()
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1.mean() - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2.mean() - target_qs).pow_(2).mean()

        critic_update_start = time()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).mean().backward(retain_graph=False)

        self.optim_critic.step()

        critic_update_end = time()
        print(f"critic backward time is {critic_update_end - critic_update_start}")


    def update_actor(self, states, goal, env):
        actions, log_pis = self.explore(states, goal, self.batch_size)
        next_states = torch.empty((self.batch_size, states[0].shape[0], states[0].shape[1]), dtype=torch.float, device=self.device)

        act_start = time()
        for i in range(self.batch_size):
            next_states[i] = env.action(states[i], [actions[i][0], actions[i][1]], env.cutter[actions[i][2]], actions[i][3])

        act_end = time()
        print(f"act time is {act_end - act_start}")

        qs = self.critic(states, next_states, goal, batch_size = self.batch_size)
        loss_actor = (self.alpha * torch.tensor(log_pis).mean() - torch.min(qs, dim=-1).values.mean()).mean()

        actor_update_start = time()
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)

        self.optim_actor.step()
        actor_update_end = time()
        print(f"actor backward time is {actor_update_end - actor_update_start}")

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)     #パラメータtを引数倍する
            t.data.add_(self.tau * s.data)  #パラメータtに引数分の加減をおこなう