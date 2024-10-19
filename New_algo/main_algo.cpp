#include <iostream>
#include <vector>
using namespace std;


vector<vector<int>> board, goal;
vector<bool> num_apper = [false, false, false, false]

void main_algo() {
	bool chain_eq = true;
	for (int y = 0; y < board[0].size(); y++) {
		for (int x = 0; x < board.size(); x++) {

			if (board[y][x] == goal[y][x] and chain_eq){
				//左上から確認していき、連続している間はpass
				break;
			}

			//違う箇所が現れたらそれ以降のpieceから同じ値を探索
			chain_eq = false;
			num_apper[board[y][x]] = true;

			goal_piece = goal[y][x];
			int column_num = 0;
			for(int y_sel = y + 1; y_sel<board[0].size(); y_sel++){
				column_num++;
				
				int raw_num = 0;
				for (int x_sel = x + 1; x_sel < board.size(); x_sel++) {
					raw_num++;
					if goal_piece == board[y_sel][x_sel]{
						int column_cut_size = 0;
						
					    //
					}

				}
			}
		}
	}
}

int main() {
	board, goal = //通信およびboard取得用の外部クラスを作成

	
}
