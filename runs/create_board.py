"""
goalのboardを予め設定してそれに対してcutterの1の間隔に沿って端からピースを移動させる
これによって確実にgoalへたどり着けることが保証されたboardを作成できる
もしかしたらこれを利用すれば本番でも、データのgoalに関してのデータを生成でき、学習に使用できるかも、、（時間制限的に厳しい）

例；
[                   
[1,2,3,1,2,3], 
[1,2,3,1,2,3],
[3,2,1,3,2,1],
[3,2,1,3,2,1],
[3,2,1,3,2,1]
]

(0,0)に[[1,0]*2]の抜型を使用して左に動かして上記のboardになったと考えて逆算する
そうすると、右の3,3を前に持ってくればよいことがわかるのでboardは、

[                   
[3,1,2,3,1,2], 
[3,1,2,3,1,2],
[3,2,1,3,2,1],
[3,2,1,3,2,1],
[3,2,1,3,2,1]
]

となる。
ー＞(0,0)に[[1,0]*2]の抜型を使用して左に動かして上記のboardになったと考えて逆算する
この部分は乱数で決定すればよいので、
ー＞右の3,3を前に持ってくればよいことがわかる
これを可能とするアルゴリズムを実装する

数字を選ぶ向きは、仮定した行動の逆（上記の例なら、左に動かすので、選ぶのは右）、選ぶ数字の個数は抜き型の行（列）にある1の数
(x,y)に抜型を適応するとすると、選んだ数字が配置されるindexは（ x+(cutterの1のx_index), y+(cutterの1のy_index ）についての各数字に対応する組み合わせ
"""

import random
import copy
from Env import transition

def create_train_board(seed=0, board_shape=[32, 32], cutter_add_num=0, num_of_shuffle=5, goal = None):
  random.seed(seed)

  goal_board = []
  actions = []

  for i in range(board_shape[0]):
    goal_board.append([])

    for j in range(board_shape[1]):
      goal_board[i].append(random.randint(0, 3))

  if goal is not None:
    goal_board = copy.deepcopy(goal)

  cutter = create_cutter(cutter_add_num)

  start_board = copy.deepcopy(goal_board)

  """
    以下でactionをもとにしたstart_boardの成形を行う
  """
  for i in range(num_of_shuffle):
    random.seed(seed + i)

    X = random.randint(0, board_shape[0] - 1)
    Y = random.randint(0, board_shape[1] - 1)

    cutter_sellect_number = random.randint(0, len(cutter) - 1)
    print(cutter_sellect_number)
    tmp_cutter = cutter[cutter_sellect_number]
    use_cutter_shape = [0, 0]

    if board_shape[0] - X < len(tmp_cutter):
      use_cutter_shape[0] = board_shape[0] - X
    else:
      use_cutter_shape[0] = len(tmp_cutter)

    if board_shape[1] - Y < len(tmp_cutter[0]):
      use_cutter_shape[1] = board_shape[1] - Y
    else:
      use_cutter_shape[1] = len(tmp_cutter[0])

    use_cutter = []
    for i in range(use_cutter_shape[0]):
      use_cutter.append([])
      for j in range(use_cutter_shape[1]):
        use_cutter[i].append(tmp_cutter[i][j])

      print(use_cutter[i])


    direct = random.randint(0, 3)

    actions.append([X, Y, cutter_sellect_number, direct])

    print(f"X={X}, Y={Y}, cutter's shape = ({len(use_cutter)}, {len(use_cutter[0])}), direct={direct}")

    cutter_one_nums = []

    if direct in [0, 1]:
      # 上下方向に動く場合にcutterの列ごとの1の枚数を調べる
      for i in range(len(use_cutter[0])):
        count = 0
        for j in range(len(use_cutter)):
          if use_cutter[j][i] == 1:
            count += 1

        cutter_one_nums.append(count)

    elif direct in [2, 3]:
      # 左右方向に動く場合にcutterの列ごとの1の枚数を調べる
      for i in range(len(use_cutter)):
        count = 0
        for j in range(len(use_cutter[0])):
          if use_cutter[i][j] == 1:
            count += 1

        cutter_one_nums.append(count)


    if direct == 0:
      # 上方向へ移動
      # ->下側のピースを選択して間に入れ込む

      cut_pieces = []
      for i in range(len(use_cutter[0])):
        cut_pieces.append([])
        for j in range(board_shape[0] - cutter_one_nums[i], board_shape[0]):
          # 抜き取るピース（抜き取った結果移動してきたピース）を取得し、その箇所のピースを穴あきとする
          cut_pieces[i].append(start_board[j][Y+i])
          start_board[j][Y+i] = -1

      # 切り取られず残るピースを保存（cut_piecesを間に埋め込むので一時保存）
      rem_pieces = []
      for i in range(len(use_cutter[0])):
        rem_pieces.append([])
        for j in range(X, board_shape[0] - cutter_one_nums[i]):
          if start_board[j][Y+i] != -1:
            rem_pieces[i].append(start_board[j][Y+i])
            start_board[j][Y+i] = -1

      for i in range(len(use_cutter[0])):
        cut_piece_counter = 0
        rem_piece_counter = 0
        for j in range(X, board_shape[0]):

          cutter_index = j - X
          if len(use_cutter) > cutter_index and use_cutter[cutter_index][i] == 1:
            start_board[j][Y+i] = cut_pieces[i][cut_piece_counter]
            cut_piece_counter += 1

          else:
            start_board[j][Y+i] = rem_pieces[i][rem_piece_counter]
            rem_piece_counter += 1

    elif direct == 1:
      # 下方向へ移動

      cut_pieces = []
      for i in range(len(use_cutter[0])):
        cut_pieces.append([])
        for j in range(cutter_one_nums[i]):
          # 抜き取るピース（抜き取った結果移動してきたピース）を取得し、その箇所のピースを穴あきとする
          cut_pieces[i].append(start_board[j][Y+i])
          start_board[j][Y+i] = -1

      # 切り取られず残るピースを保存（cut_piecesを間に埋め込むので一時保存）
      rem_pieces = []
      for i in range(len(use_cutter[0])):
        rem_pieces.append([])
        for j in range(cutter_one_nums[i], X + len(use_cutter)):
          if start_board[j][Y+i] != -1:
            rem_pieces[i].append(start_board[j][Y+i])
            start_board[j][Y+i] = -1

      for i in range(len(use_cutter[0])):
        cut_piece_counter = 0
        rem_piece_counter = 0
        for j in range(0, X + len(use_cutter)):

          if j-X >= 0 and use_cutter[j - X][i] == 1:
            start_board[j][Y+i] = cut_pieces[i][cut_piece_counter]
            cut_piece_counter += 1

          else:
            start_board[j][Y+i] = rem_pieces[i][rem_piece_counter]
            rem_piece_counter += 1

    elif direct == 2:
      # 左方向へ移動

      cut_pieces = []
      for i in range(len(use_cutter)):
        cut_pieces.append([])
        for j in range(board_shape[1] - cutter_one_nums[i], board_shape[1]):
          # 抜き取るピース（抜き取った結果移動してきたピース）を取得し、その箇所のピースを穴あきとする
          cut_pieces[i].append(start_board[X+i][j])
          start_board[X+i][j] = -1


      # 切り取られず残るピースを保存（cut_piecesを間に埋め込むので一時保存）
      rem_pieces = []
      for i in range(len(use_cutter)):
        rem_pieces.append([])
        for j in range(Y, board_shape[1] - cutter_one_nums[i]):
          if start_board[X+i][j] != -1:
            rem_pieces[i].append(start_board[X+i][j])
            start_board[X+i][j] = -1
      for i in range(len(use_cutter)):
        cut_piece_counter = 0
        rem_piece_counter = 0
        for j in range(Y, board_shape[1]):

          cutter_index = j - Y
          if len(use_cutter[0]) > cutter_index and use_cutter[i][cutter_index] == 1:
            start_board[X+i][j] = cut_pieces[i][cut_piece_counter]
            cut_piece_counter += 1

          else:
            start_board[X+i][j] = rem_pieces[i][rem_piece_counter]
            rem_piece_counter += 1

    elif direct == 3:
      # 右方向へ移動

      cut_pieces = []
      for i in range(len(use_cutter)):
        cut_pieces.append([])
        for j in range(cutter_one_nums[i]):
          # 抜き取るピース（抜き取った結果移動してきたピース）を取得し、その箇所のピースを穴あきとする
          cut_pieces[i].append(start_board[X+i][j])
          start_board[X+i][j] = -1

      print(cut_pieces)

      # 切り取られず残るピースを保存（cut_piecesを間に埋め込むので一時保存）
      rem_pieces = []
      for i in range(len(use_cutter)):
        rem_pieces.append([])
        for j in range(cutter_one_nums[i], Y +  len(use_cutter[0])):
          if start_board[X+i][j] != -1:
            rem_pieces[i].append(start_board[X+i][j])
            start_board[X+i][j] = -1

      print(rem_pieces)

      for i in range(len(use_cutter)):
        cut_piece_counter = 0
        rem_piece_counter = 0
        for j in range(0, Y + len(use_cutter[0])):

          cutter_index = j - Y
          if 0 <= cutter_index and use_cutter[i][cutter_index] == 1:
            start_board[X+i][j] = cut_pieces[i][cut_piece_counter]
            cut_piece_counter += 1

          else:
            start_board[X+i][j] = rem_pieces[i][rem_piece_counter]
            rem_piece_counter += 1

  return start_board, goal_board, cutter, actions

def create_cutter(add_num=0):
  # 定型抜き型を作成
  cutter = [[[1]]]

  for size in [2, 4, 8, 16, 32, 64, 128, 256]:
    grid = []
    # すべてが1の抜き型
    for i in range(size):
      grid.append([])
      for j in range(size):
        grid[i].append(1)

    cutter.append(grid)

    grid = []
    # 1マスおきに列が1の抜き型
    for i in range(size):
      grid.append([])
      for j in range(size):
        if j % 2 == 0:
          grid[i].append(1)
        else:
          grid[i].append(0)

    cutter.append(grid)

    grid = []
    # 1マスおきに行が1の抜き型
    for i in range(size):
      grid.append([])
      for j in range(size):
        if i % 2 == 0:
          grid[i].append(1)
        else:
          grid[i].append(0)

    cutter.append(grid)

  if add_num != 0:
    """
    add_num(追加のcutter数)に応じてrandomでcutterを作成する
    """
    pass

  return cutter

if __name__ == "__main__":

  train_board, goal_board, cutter, actions = create_train_board(seed=2,
                                                       board_shape=[32,32],
                                                       cutter_add_num=0,
                                                       num_of_shuffle=10)

  num_counterT = [0,0,0,0]

  print("train")
  for i in range(len(train_board)):
    print(train_board[i])
    for j in range(len(train_board[0])):
      num_counterT[train_board[i][j]] += 1

  num_counterG = [0,0,0,0]
  print()
  print("goal")
  for i in range(len(goal_board)):
    print(goal_board[i])
    for j in range(len(train_board[0])):
      num_counterG[goal_board[i][j]] += 1

  for i in range(4):
    if num_counterG[i] != num_counterT[i]:
      print(num_counterG)
      print(num_counterT)
      print("不一致")
      break

  env = transition(train_board, cutter, goal_board)
  for i in range(9, -1, -1):
    next_state, _, done, _ = env.step(actions[i])
    print(done)

  num_counterA = [0,0,0,0]
  print()
  print("state")
  for i in range(len(goal_board)):
    print(next_state[i])
    for j in range(len(goal_board[0])):
      num_counterA[next_state[i][j]] += 1

  for i in range(4):
    if num_counterA[i] != num_counterG[i]:
      print(num_counterG)
      print(num_counterA)
      print("不一致")