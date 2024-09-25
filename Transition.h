#ifndef TRANSITION_H
#define TRANSITION_H

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>
#include <tuple>

// Use std::vector to replace Python's list functionality
using namespace std;

class Transition {
public:
    Transition(const vector<vector<int>>& board,
               const vector<vector<vector<int>>>& cutter,
               const vector<vector<int>>& goal,
               int frequ = 1, bool test = false);

    tuple<vector<vector<int>>, int, bool, int> step(const vector<int>& act);
    vector<vector<int>> action(vector<vector<int>>& board,
                               const vector<int>& pos,
                               const vector<vector<int>>& cutter,
                               int direction);
    int reward(const vector<vector<int>>& before_state,
               const vector<vector<int>>& state);

    void seed(unsigned int seed_val);
    vector<vector<int>> reset();
    vector<int> action_sample();

private:
    vector<vector<int>> start, board, goal;
    vector<vector<vector<int>>> cutter;
    int x_boardsize, y_boardsize;
    int frequ, num_step, _max_episode_steps;
    int rew, before_rew, max_rew, max_eq, best_step, num_of_cutter;
    bool done, test;
    vector<int> action_shape;
    vector<vector<int>> ans_board;
    tuple<int, vector<tuple<int, int, int, int>>> ans;
    
    void initialize_counters();
};

// Constructor implementation
Transition::Transition(const vector<vector<int>>& board,
                       const vector<vector<vector<int>>>& cutter,
                       const vector<vector<int>>& goal,
                       int frequ, bool test) :
    start(board), board(board), goal(goal), cutter(cutter), frequ(frequ),
    test(test), done(false), num_step(0), _max_episode_steps(100), rew(0),
    before_rew(0), max_rew(-10000), max_eq(-1), best_step(0) {

    x_boardsize = board.size();
    y_boardsize = board[0].size();
    num_of_cutter = cutter.size();
    action_shape = {4};
}

// Step function
tuple<vector<vector<int>>, int, bool, int>
Transition::step(const vector<int>& act) {
    num_step++;

    vector<vector<int>> state = board;  // Copy of current board
    vector<vector<int>> next_state = action(state, {act[0], act[1]}, cutter[act[2]], act[3]);
    board = next_state;

    if (num_step % frequ == 0) {
        rew += reward(state, next_state);
    }

    if (board == goal || num_step == _max_episode_steps) {
        done = true;
    }

    if (test) {
        // If in test mode, save actions
        // Example: ans stores best result
        get<1>(ans).push_back(make_tuple(act[2], act[0], act[1], act[3]));
    }

    return make_tuple(next_state, rew, done, num_step);
}

// Action function to update the board based on the action parameters
vector<vector<int>> Transition::action(vector<vector<int>>& board,
                                       const vector<int>& pos,
                                       const vector<vector<int>>& cutter,
                                       int direction) {

    // Your optimized logic for handling the cut and shift
    // Convert nested loops to efficient C++ code with STL optimizations
    int X_SIZE = cutter.size();
    int Y_SIZE = cutter[0].size();
    int X = pos[0];
    int Y = pos[1];

    bool up = (direction == 0), down = (direction == 1), 
         left = (direction == 2), right = (direction == 3);

    // Continue the rest of the function, translating logic step by step
    // with an aim to improve performance by avoiding redundant calculations
    vector<vector<int>> cut_val(256, vector<int>(256, -1));

    // Similar loop structure as Python, adjusted for C++
    // ...

    return board;  // Updated board state after action
}

// Reward calculation
int Transition::reward(const vector<vector<int>>& before_state,
                       const vector<vector<int>>& state) {
    int H = state.size(), W = state[0].size();
    int this_rew = 0, num_eq = 0, num_eq_before = 0;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (state[i][j] == goal[i][j]) num_eq++;
            if (state[i][j] == before_state[i][j]) num_eq_before++;
        }
    }

    if (num_eq == H * W) {
        this_rew += 1000;
    } else {
        this_rew += (num_eq / (H * W)) * 1000;
        if (num_eq_before == H * W) this_rew = -1000;
    }

    if (test && this_rew > max_rew) {
        max_rew = this_rew;
        max_eq = num_eq;
        best_step = num_step;
        ans_board = state;
    }

    return this_rew;
}

// Set random seed
void Transition::seed(unsigned int seed_val) {
    srand(seed_val);
}

// Reset the board to the initial state
vector<vector<int>> Transition::reset() {
    board = start;
    num_step = 0;
    done = false;
    rew = 0;
    max_rew = -10000;
    max_eq = -1;
    best_step = 0;
    return board;
}

// Random action generation
vector<int> Transition::action_sample() {
    vector<int> action(4);
    action[0] = rand() % x_boardsize;
    action[1] = rand() % y_boardsize;
    action[2] = rand() % num_of_cutter;
    action[3] = rand() % 4;
    return action;
}

#endif  // TRANSITION_H
