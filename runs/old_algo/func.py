import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
from time import sleep

import torch


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

def plot_board(board):
        # 0~3の値に対応する色を定義
        cmap = ListedColormap(['red', 'green', 'blue', 'yellow'])

        # 図を描画
        plt.imshow(board, cmap=cmap, interpolation='none')

        # カラーバーを表示して、各色が何の値に対応するかを表示
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['0', '1', '2', '3'])  # ラベルを設定

        # グリッド線を追加
        plt.grid(False)  # グリッドを非表示にする場合はTrueをFalseに変更

        # 図を表示
        plt.show()

def concat_jsonfile(filename_old, filename_new):

    send_solution = {"n": 0, "ops":[]}


    with open(filename_old, 'r') as f:
        solution_old = json.load(f)
           
    with open(filename_new, 'r') as f:
        solution_new = json.load(f)

    send_solution["n"] = solution_old["n"] + solution_new["n"]

    print("before steps", solution_old["n"])
    print("now steps", solution_new["n"])
    print("new steps", send_solution["n"])

    send_solution["ops"] = solution_old["ops"] + solution_new["ops"]

    print("ops num", len(send_solution["ops"]))

    sleep(10)

    return send_solution

    