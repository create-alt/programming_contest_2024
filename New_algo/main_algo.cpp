#include <iostream>
#include <vector>
using namespace std;

// json用ライブラリの読み込みを行い、下記を使用可能に
// 使用できるようになるまではvectorで代替
//  json send_json = {{"n",0},
//                    {"ops", {}}};

int n = 0; // 手数

vector<vector<int>> board, goal;
vector<vector<int>> ops;

vector<int> main_algo()
{
    vector<int> act(4);

    bool chain_eq = true;
    int pos_x, pos_y, cutter, direct;
    for (int y = 0; y < board[0].size(); y++)
    {
        // board全体に対してサーチ
        for (int x = 0; x < board.size(); x++)
        {

            if (board[y][x] == goal[y][x] and chain_eq)
            {
                // 左上から確認していき、連続している間はpass
                break;
            }

            // 違う箇所が現れたらそれ以降のpieceから同じ値を探索
            chain_eq = false;

            int goal_piece = goal[y][x];
            int column_num = 0; // 列方向の移動数

            //(x,y)に寄せるピースを探索
            for (int y_sel = y; y_sel < board[0].size(); y_sel++)
            {

                int raw_num = 0; // 行方向の移動数
                for (int x_sel = x + 1; x_sel < board.size(); x_sel++)
                {
                    raw_num++;

                    // 寄せるピースが見つかったらそのピースを正しく移動させるための行動を選択
                    if (goal_piece == board[y_sel][x_sel])
                    {
                        // int column_cut_size = 0;

                        act[0] = 0;            // cutterの種類（ひとまず簡単のため[[1]]で考える）
                        act[1], act[2] = x, y; // 抜き型の起点となる座標（できれば負の数から適応も考える）

                        for (int i = 0; i < column_num; i++)
                        {
                            act[3] = 0; // 移動方向：上
                            n++;        // 手数を更新

                            ops.push_back(act); // vector: opsに行動を記録
                            action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                        }

                        for (int i = 0; i < raw_num; i++)
                        {
                            act[3] = 2; // 移動方向：左
                            n++;        // 手数を更新

                            ops.push_back(act); // vector: opsに行動を記録
                            action(board, act); // 選択した行動を行いboardを更新（boardは参照渡し）
                        }
                    }
                }
                column_num++;
            }
        }
    }
}

int main()
{
    board, goal = // 通信およびboard取得用の外部クラスを作成

    main_algo() // アルゴリズムを実行して行動を取得

    /*
    以下で、jsonファイルの作成と提出を行う。
    nとopsはグローバル変数
    */
}
