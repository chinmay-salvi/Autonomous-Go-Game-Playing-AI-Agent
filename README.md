# Autonomous-Go-Game-Playing-AI-Agent (5x5 Go Game)

This project implements an Artificial Intelligence agent designed to play the game of Go on a 5x5 board. The AI leverages the **Minimax algorithm with Alpha-Beta pruning** to determine the optimal moves and includes various strategies such as move ordering and time management to compete effectively.

## Project Structure

*   **`my_player3.py`**: The core AI implementation. It contains the `AlphaBetaMaxMinPlayer` which utilizes Minimax with Alpha-Beta pruning to select the best moves.
*   **`host.py`**: The game engine and host environment. It manages the board state, enforces the rules of Go (including the KO rule and liberties), and acts as the judge for the game.
*   **`random_player.py`**: A baseline AI agent that makes valid moves entirely at random.
*   **`build.sh`**: A shell script used to compile and run the AI agents against each other to evaluate performance.

## How the AI Works

The AI is built on the `AlphaBetaMaxMinPlayer` class and employs several key strategies to maximize its win rate on the 5x5 board:

### 1. Minimax with Alpha-Beta Pruning
The agent explores the game tree to a certain depth (default `max_depth = 5`) to predict the opponent's responses and find the sequence of moves that maximizes its own score while minimizing the opponent's. Alpha-Beta pruning is used to cut off branches that don't need to be explored, significantly speeding up the search process.

### 2. Heuristic Evaluation Function
When the maximum depth is reached or time runs out, the agent evaluates the board state using a heuristic function. The evaluation calculates the material advantage by comparing the number of stones the AI has on the board versus the opponent, factoring in the *Komi* (compensation points for the second player).

### 3. Move Ordering
To improve the efficiency of Alpha-Beta pruning, the agent sorts the possible moves before exploring them. Moves are prioritized based on:
1.  **Liberties:** Moves that result in more liberties.
2.  **Capture Potential:** Moves that capture the opponent's stones.
3.  **Centrality:** Moves closer to the center of the 5x5 board (Manhattan distance from the center).

### 4. Dynamic Depth & Time Management
The agent has a strict time limit of 9.6 seconds per move. It monitors the elapsed time during the search and immediately returns the best move found so far if the limit is approaching. Additionally, as the game progresses and fewer empty spots remain (specifically after 14 total moves), the agent dynamically increases its search depth to 7 to calculate the endgame more precisely.

## Running the Agent

You can use the provided `build.sh` script to run matches between your AI agent and the baseline `random_player`. 

```bash
cd Code
./build.sh
```

The script will clean up old files, run the match, and output the winner of each round.
