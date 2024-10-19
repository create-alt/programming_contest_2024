#include <vector>
#include <algorithm>
#include <map>
#include <string>
using namespace std;

vector<vector<int>> action(vector<vector<int>> &board, vector<int> &act, int batch_size = 1)
{

    std::vector<std::vector<std::vector<int>>> cutter = {{{1}}};

    // サイズリスト
    std::vector<int> sizes = {2, 4, 8, 16, 32, 64, 128, 256};

    // サイズごとの定型抜き型を作成
    for (int size : sizes)
    {
        std::vector<std::vector<int>> grid;

        // すべてが1の抜き型
        grid.clear();
        for (int i = 0; i < size; ++i)
        {
            std::vector<int> row(size, 1);
            grid.push_back(row);
        }
        cutter.push_back(grid);

        // 1マスおきに列が1の抜き型
        grid.clear();
        for (int i = 0; i < size; ++i)
        {
            std::vector<int> row;
            for (int j = 0; j < size; ++j)
            {
                row.push_back((j % 2 == 0) ? 1 : 0);
            }
            grid.push_back(row);
        }
        cutter.push_back(grid);

        // 1マスおきに行が1の抜き型
        grid.clear();
        for (int i = 0; i < size; ++i)
        {
            std::vector<int> row;
            for (int j = 0; j < size; ++j)
            {
                row.push_back((i % 2 == 0) ? 1 : 0);
            }
            grid.push_back(row);
        }
        cutter.push_back(grid);
    }

    // 一般的な抜き型をcutterに追加する
    // std::map<std::string, std::map<std::string, std::vector<std::vector<int>>>> data;

    // `data`には、一般的な抜き型のパターンが格納されていると仮定
    // std::vector<std::vector<int>> cutters = data["general"]["patterns"];

    // for (int num = 0; num < data["general"]["n"]; ++num) {
    //     auto cut_info = cutters[num];
    //     std::vector<std::vector<int>> grid;
    //     for (int i = 0; i < cut_info.size(); ++i) {
    //         std::vector<int> row;
    //         for (int j = 0; j < cut_info[i].size(); ++j) {
    //             row.push_back(cut_info[i][j]);
    //         }
    //         grid.push_back(row);
    //     }
    //     cutter.push_back(grid);
    // }

    // 使用するカッターを選択

    vector<vector<int>> use_cutter = cutter[act[0]];

    // cutterのサイズと型抜き座標を定数として格納
    int X_SIZE = use_cutter[0].size();
    int Y_SIZE = use_cutter.size();
    int X = static_cast<int>(act[1]);
    int Y = static_cast<int>(act[2]);
    int x_boardsize = board[0].size();
    int y_boardsize = board.size();

    int direction = act[3];

    // cut_valを動的に確保 (メモリの無駄を避けるため、必要サイズだけ確保)
    vector<int> cut_val;

    // 抜きピースを保存し、穴を空ける
    for (int i = 0; i < X_SIZE; ++i)
    {
        for (int j = 0; j < Y_SIZE; ++j)
        {
            if (0 <= X + i && X + i < x_boardsize && 0 <= Y + j && Y + j < y_boardsize && use_cutter[i][j] == 1)
            {
                cut_val.push_back(board[Y + j][X + i]);
                board[Y + j][X + i] = -1;
            }
        }
    }

    // direction (int) => 型抜き後移動する方向(0:上 1:下 2:左 3:右)

    // board全体の走査を省略するため、カッターの影響を受けた範囲のみ移動処理を実施
    if (direction == 0 || direction == 1)
    { // 上下方向の移動
        for (int j = X; j < X + X_SIZE - 1; ++j)
        {
            if (0 <= j && j < y_boardsize)
            {
                int count = 0;
                if (direction == 0)
                { // 上方向
                    for (int i = Y; i < y_boardsize; ++i)
                    {
                        if (board[i][j] == -1)
                        {
                            ++count;
                        }
                        else if (count > 0)
                        {
                            board[i - count][j] = board[i][j];
                            board[i][j] = -1;
                        }
                    }
                }
                else
                { // 下方向
                    for (int i = min(Y + Y_SIZE - 1, y_boardsize - 1); i >= 0; --i)
                    {
                        if (board[j][i] == -1)
                        {
                            ++count;
                        }
                        else if (count > 0)
                        {
                            board[j][i + count] = board[j][i];
                            board[j][i] = -1;
                        }
                    }
                }
            }
        }
    }
    else if (direction == 2 || direction == 3)
    { // 左右方向の移動
        for (int i = Y; i < min(Y + Y_SIZE, y_boardsize); ++i)
        {
            if (0 <= i && i < x_boardsize)
            {
                int count = 0;
                if (direction == 2)
                { // 左方向
                    for (int j = X; j < x_boardsize; ++j)
                    {
                        if (board[i][j] == -1)
                        {
                            ++count;
                        }
                        else if (count > 0)
                        {
                            board[i][j - count] = board[i][j];
                            board[i][j] = -1;
                        }
                    }
                }
                else
                { // 右方向
                    for (int j = min(X + X_SIZE - 1, x_boardsize - 1); j >= 0; --j)
                    {
                        if (board[i][j] == -1)
                        {
                            ++count;
                        }
                        else if (count > 0)
                        {
                            board[i][j + count] = board[i][j];
                            board[i][j] = -1;
                        }
                    }
                }
            }
        }
    }

    // 元の位置にcut_valを戻す (全探索を避け、ピースが置かれた場所にのみ戻す)
    int cut_index = 0;
    for (int i = 0; i < X_SIZE; ++i)
    {
        for (int j = 0; j < Y_SIZE; ++j)
        {
            if (cut_index < cut_val.size())
            {
                if (direction == 0)
                {
                    if (0 <= x_boardsize - X_SIZE + i && x_boardsize - X_SIZE + i < x_boardsize && 0 <= Y + j && Y + j < y_boardsize && board[Y + j][x_boardsize - X_SIZE + i] == -1)
                    {
                        board[Y + j][x_boardsize - X_SIZE + i] = cut_val[cut_index++];
                    }
                }
                else if (direction == 1)
                {
                    if (0 <= i && i < x_boardsize && 0 <= Y + j && Y + j < y_boardsize && board[Y + j][i] == -1)
                    {
                        board[Y + j][i] = cut_val[cut_index++];
                    }
                }
                else if (direction == 2)
                {
                    if (0 <= X + i && X + i < x_boardsize && 0 <= y_boardsize - Y_SIZE + j && y_boardsize - Y_SIZE + j < y_boardsize && board[y_boardsize - Y_SIZE + j][X + i] == -1)
                    {
                        board[y_boardsize - Y_SIZE + j][X + i] = cut_val[cut_index++];
                    }
                }
                else if (direction == 3)
                {
                    if (0 <= X + i && X + i < x_boardsize && 0 <= j && j < y_boardsize && board[j][X + i] == -1)
                    {
                        board[j][X + i] = cut_val[cut_index++];
                    }
                }
            }
        }
    }

    return board;
}
