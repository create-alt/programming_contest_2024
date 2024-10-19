#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <string>

// https://github.com/nlohmann/json より引用
#include "json.hpp"
using json = nlohmann::json;
using namespace std;

std::vector<std::vector<std::vector<int>>> cutter = {{{1}}};
int use_cutter_basic_number = 19; // ここをいじると全一致しなくなる場合があるので注意

void print_board(vector<vector<int>> &grid)
{
    for (int i = 0; i < grid.size(); i++)
    {
        for (int j = 0; j < grid[0].size(); j++)
        {
            cout << grid[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<vector<int>> action(vector<vector<int>> &board, vector<int> &act)
{
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

    // cout << X_SIZE << " " << Y_SIZE << " " << endl;
    // cout << X << " " << Y << endl;
    // cout << x_boardsize << " " << y_boardsize << endl;
    // cout << direction << endl;

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
        for (int j = X; j < min(X + X_SIZE, x_boardsize); ++j)
        {
            if (0 <= j)
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
                        if (board[i][j] == -1)
                        {
                            ++count;
                        }
                        else if (count > 0)
                        {
                            board[i + count][j] = board[i][j];
                            board[i][j] = -1;
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
            if (0 <= i)
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
    // もともと左から縦方向にピースを抜き取っているので縦方向にもどしていく必要がある
    int cut_index = 0;
    for (int j = 0; j < X_SIZE; ++j)
    {
        for (int i = 0; i < Y_SIZE; ++i)
        {
            if (cut_index < cut_val.size())
            {
                if (direction == 0)
                {
                    if (0 <= X + j && X + j < x_boardsize && 0 <= y_boardsize - Y_SIZE + i && y_boardsize - Y_SIZE + i < y_boardsize && board[y_boardsize - Y_SIZE + i][X + j] == -1)
                    {
                        board[y_boardsize - Y_SIZE + i][X + j] = cut_val[cut_index++];
                    }
                }
                else if (direction == 1)
                {
                    if (0 <= X + j && X + j < x_boardsize && 0 <= Y && Y < y_boardsize && board[Y][X + j] == -1)
                    {
                        board[Y][X + j] = cut_val[cut_index++];
                    }
                }
            }
        }
    }

    for (int j = 0; j < X_SIZE; ++j)
    {
        for (int i = 0; i < Y_SIZE; ++i)
        {
            if (cut_index < cut_val.size())
            {
                if (direction == 2)
                {
                    if (0 <= x_boardsize - X_SIZE + j && x_boardsize - X_SIZE + j < x_boardsize && 0 <= Y + i && Y + i < y_boardsize && board[Y + i][x_boardsize - X_SIZE + j] == -1)
                    {
                        board[Y + i][x_boardsize - X_SIZE + j] = cut_val[cut_index++];
                    }
                }
                else if (direction == 3)
                {
                    if (0 <= X + j && X + j < x_boardsize && 0 <= Y + i && Y + i < y_boardsize && board[Y + i][j] == -1)
                    {
                        board[Y + i][j] = cut_val[cut_index++];
                    }
                }
            }
        }
    }

    return board;
}

// json用ライブラリの読み込みを行い、下記を使用可能に
// 使用できるようになるまではvectorで代替
//  json send_json = {{"n",0},
//                    {"ops", {}}};

int n = 0; // 手数

vector<vector<int>> board, goal;
vector<vector<int>> ops;

void main_algo()
{
    vector<int> act(4);

    int pos_x, pos_y, cutter, direct;

    for (int y = 0; y < board.size(); y++)
    {

        // board全体に対してサーチ
        for (int x = 0; x < board[0].size(); x++)
        {

            if (board[y][x] == goal[y][x])
            {
                // 左上から確認していき、連続している間はpass
                continue;
            }

            int goal_piece = goal[y][x];
            // cout << "goal : " << goal_piece;
            // cout << board[y][x] << "x: " << x << " y: " << y << endl;

            //(x,y)に寄せるピースを探索

            int tmp_x = 0;
            int tmp_y = 0;
            int before_dis = 1000;
            int count = 0;
            int use_x;
            int use_y;

            for (int y_sel = y; y_sel < board.size(); y_sel++)
            {

                // 本二重ループではあくまで、移動させるピース一つを探索している
                bool quit = false;
                bool first = true;

                for (int x_sel = 0; x_sel < board[0].size(); x_sel++)
                {

                    if (y_sel == y && x_sel <= x)
                        continue;

                    // 今のところ動かなくなるが、修正すればもしかしたら、、
                    // 以下は幅優先にすることで無駄な計算を省ける
                    for (int i = y_sel; i < board.size(); i++)
                    {
                        for (int j = 0; j < board[0].size(); j++)
                        {
                            if (i == y_sel && j <= x_sel)
                                continue;
                            if (goal_piece == board[i][j])
                            {
                                int now_dis = abs(i - y_sel) + abs(j - x_sel);
                                if (now_dis < before_dis)
                                {
                                    before_dis = now_dis;
                                    use_x = j;
                                    use_y = i;
                                }
                            }
                        }
                    }

                    // use_x = x_sel;
                    // use_y = y_sel;

                    // 寄せるピースが見つかったらそのピースを正しく移動させるための行動を選択
                    if (goal_piece == board[use_y][use_x])
                    {

                        // int column_cut_size = 0;

                        act[0] = 0; // cutterの種類（ひとまず簡単のため[[1]]で考える）

                        if (use_x < x)
                        {
                            act[1] = x;
                            act[2] = use_y;
                            act[3] = 3;

                            // cout << "1act is " << act[0] << " " << act[1] << " " << act[2] << " " << act[3] << endl;
                            // cout << use_x << " " << use_y << endl;
                            // cout << "raw_num is " << x - use_x << endl;

                            int move_length = x - use_x;
                            int times = 0;
                            for (int i = 1; i < 128; i *= 2)
                            {
                                if (move_length >= 128 / i)
                                {
                                    act[0] = use_cutter_basic_number - 3 * times;
                                    n++;
                                    ops.push_back(act);         // vector: opsに行動を記録
                                    board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                                    move_length -= 128 / i;     // ここにcutterがboardからはみ出た時の処理が必要

                                    if ((x + (128 / i)) >= board[0].size())
                                    {
                                        move_length += (x + (128 / i)) - board[0].size();
                                    }
                                }
                                times++;
                            }

                            for (int _ = 0; _ < move_length; _++)
                            {
                                act[0] = 0;

                                n++;
                                ops.push_back(act);         // vector: opsに行動を記録
                                board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                            }
                        }

                        int move_length2 = use_y - y;

                        if (move_length2 > 0)
                        {
                            // cout << use_y << endl;
                            act[1] = x; // 抜き型の起点となる座標（できれば負の数から適応も考える）
                            act[2] = y; // 抜き型の起点となる座標（できれば負の数から適応も考える）
                            act[3] = 0; // 移動方向：上

                            if (x < use_x)
                                act[1] = use_x;

                            int times = 0;
                            for (int i = 1; i < 128; i *= 2)
                            {

                                if (move_length2 >= 128 / i)
                                {
                                    act[0] = use_cutter_basic_number - 3 * times;
                                    n++;
                                    ops.push_back(act);         // vector: opsに行動を記録
                                    board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                                    move_length2 -= 128 / i;

                                    if ((y + (128 / i)) >= board.size())
                                    {
                                        move_length2 += (y + (128 / i)) - board.size();
                                    }
                                }
                                times++;
                            }

                            for (int _ = 0; _ < move_length2; _++)
                            {
                                act[0] = 0;
                                ops.push_back(act);         // vector: opsに行動を記録
                                board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                                // cout << "2act is " << act[0] << " " << act[1] << " " << act[2] << " " << act[3] << endl;
                                // cout << use_x << " " << use_y << endl;
                            }
                        }

                        int move_length3 = use_x - x;

                        if (move_length3 > 0)
                        {
                            act[1] = x; // 抜き型の起点となる座標（できれば負の数から適応も考える）
                            act[2] = y; // 抜き型の起点となる座標（できれば負の数から適応も考える）
                            act[3] = 2; // 移動方向：左

                            int times = 0;
                            for (int i = 1; i < 128; i *= 2)
                            {

                                if (move_length3 >= 128 / i)
                                {
                                    act[0] = use_cutter_basic_number - 3 * times;
                                    n++;
                                    ops.push_back(act);         // vector: opsに行動を記録
                                    board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                                    move_length3 -= 128 / i;

                                    if ((x + (128 / i)) >= board[0].size())
                                    {
                                        move_length3 += (x + (128 / i)) - board[0].size();
                                    }
                                }
                                times++;
                            }
                            for (int _ = 0; _ < move_length3; _++)
                            {
                                act[0] = 0;
                                ops.push_back(act);         // vector: opsに行動を記録
                                board = action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）

                                // cout << "3act is " << act[0] << " " << act[1] << " " << act[2] << " " << act[3] << endl;
                                // cout << use_x << " " << use_y << endl;
                            }
                        }

                        // cout << endl;
                        // print_board(board);

                        quit = true;
                        break;
                    }
                }
                if (quit)
                    break;
            }
        }
    }
};

int main()
{
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
                row.push_back((i % 2 == 0) ? 1 : 0);
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
                row.push_back((j % 2 == 0) ? 1 : 0);
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

    // https://qiita.com/YukkuriAzuki/items/b856811b9a5b012d4907 参照
    json recieve_json;

    string filename = "response.json";

    // ファイルを読み込んで内容を画面に表示する
    // 読み込みに失敗した場合はエラーを表示する
    ifstream ifs(filename.c_str());
    if (ifs.good())
    {
        ifs >> recieve_json;
    }
    else
    {
        cout << "ファイルの読み込みに失敗しました" << endl;
    }

    vector<string> get_board = recieve_json["board"]["start"];

    vector<string> get_goal = recieve_json["board"]["goal"];

    for (int i = 0; i < get_board.size(); i++)
    {
        board.push_back({});
        goal.push_back({});
        for (int j = 0; j < get_board[0].size(); j++)
        {
            board[i].push_back(int(get_board[i][j] - '0'));
            goal[i].push_back(int(get_goal[i][j] - '0'));
        }
    }

    int eq = 0;
    for (int i = 0; i < board.size(); i++)
    {
        for (int j = 0; j < board[0].size(); j++)
        {
            if (board[i][j] == goal[i][j])
            {
                eq++;
            }
        }
    }

    cout << "default eq is " << eq << endl;

    main_algo(); // アルゴリズムを実行して行動を取得

    eq = 0;
    for (int i = 0; i < board.size(); i++)
    {
        for (int j = 0; j < board[0].size(); j++)
        {
            cout << board[i][j] << " ";

            if (board[i][j] == goal[i][j])
            {

                eq++;
            }
        }

        cout << endl;
    }

    cout << "after search's eq nums is " << eq << endl;
    cout << "n is " << n << endl;

    /*
    以下で、jsonファイルの作成と提出を行う。
    nとopsはグローバル変数
    */

    json save_file;

    // {
    // "n": 1,
    // "ops": [
    //     {
    //     "p": 12,
    //     "x": 0,
    //     "y": 0,
    //     "s": 3
    //     }
    //   ]
    // }

    save_file["n"] = n;
    for (int i = 0; i < n; i++)
    {
        json json_object = {
            {"p", ops[i][0]},
            {"x", ops[i][1]},
            {"y", ops[i][2]},
            {"s", ops[i][3]}};

        save_file["ops"].push_back(json_object);
    }

    // JSONオブジェクトをファイルに書き出し
    ofstream file("solution.json");
    if (file.is_open())
    {
        file << save_file.dump(4); // インデント幅4で成形
        file.close();
        cout << "done creating json file" << endl;
    }
    else
    {
        cerr << "cannot open the file" << endl;
    }
}
