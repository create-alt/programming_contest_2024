from time import time
import random
import copy

#2重ループが頻出しており、実行時間が長いと考えられるので要改善
class transition():
    """
    本クラスはenv(学習環境)として扱うクラスであるので、
    行動に対して次の状態と報酬、実行が終わったかどうかを返す。
    """
    def __init__(self, board, cutter, goal, frequ=1, test=False, EPISODE_SIZE = 1000, get_actions = None):
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

        self.before_eq = self.default_eq / (len(board) * len(board[0]))

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

        self.save_file_name   = None
        self.before_file_name = None

        self.use_actions = get_actions
        if get_actions is not None:
            self.supervised_index = len(get_actions) - 1 

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

      # cutterのサイズと型抜き座標を定数として格納
      X_SIZE, Y_SIZE = len(cutter), len(cutter[0])
      X, Y = pos[0], pos[1]
      x_boardsize, y_boardsize = len(board), len(board[0])

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
        # progress = (num_eq - self.default_eq) / total_pieces
        # this_rew = progress * 100  # 進捗に基づいた報酬
        this_rew = (num_eq / total_pieces) * 100

        # # 一致数が減少してもペナルティは軽減する
        # if num_eq < self.before_eq:
        #     this_rew -= 5  # 一時的な減少に対して軽いペナルティ

        # 全一致時の大きな報酬
        if num_eq == total_pieces:
            this_rew += 1000  # 全一致時のボーナス
            self.done = True

        # # 一致数が改善している場合は報酬を増加
        # if num_eq > self.before_eq:
        #     this_rew += 50  # 改善が見られる場合のボーナス

        # # 前回の一致数を更新
        # self.num_eq = num_eq / total_pieces
        # self.before_eq = num_eq

        # デバッグ用の表示
        if self.num_step % 100 == 0:
            print(f"{self.num_step} this_rew : {this_rew}")

        # 最良の報酬とステップの更新
        if num_eq > self.max_eq:
            self.max_rew = this_rew
            self.max_eq = num_eq
            self.best_step = self.num_step
            self.ans_board = copy.deepcopy(state)

            if self.before_file_name is None or self.max_eq > int(self.before_file_name):
                self.save_file_name = f"{self.max_eq}"

        return this_rew

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        # self.board = copy.deepcopy(self.start)

        #訓練用の処理（本当に学習に対応できているか怪しいので要検証）
        if self.ans_board is not None:
            self.board = copy.deepcopy(self.ans_board) #訓練用
        else:
            self.board = copy.deepcopy(self.start)

        self.board = copy.deepcopy(self.start)

        # self.default_eq = self.max_eq

        self.before_eq = self.max_eq

        self.before_file_name = self.save_file_name
        self.save_file_name = None
        

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

        if self.use_actions is not None:
            self.supervised_index = len(self.use_actions) - 1

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
    
    def action_sample_supervised(self):

        action = []

        if self.use_actions is not None and self.supervised_index >= 0:
            action = self.use_actions[self.supervised_index]
            self.supervised_index -= 1
            if self.supervised_index < 0:
                _ = self.reset()
        else:
            action = None

        return action