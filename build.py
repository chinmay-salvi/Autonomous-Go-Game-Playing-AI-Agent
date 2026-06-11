import os.path
import time
import numpy as np
from copy import deepcopy
from datetime import datetime
from my_player3_rl import QLearningAgent, Transition


class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        # self.previous_board = None # Store the previous board
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Intialize died pieces to be empty
        self.n_move = 0  # Trace the number of moves
        self.max_move = n * n - 1  # The max movement of a Go game
        self.komi = n / 2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player

    def init_board(self, n):
        """
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        """
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        """
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        """

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        """
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        """
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        """
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        """
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        """
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        """
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        """
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        """
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        """
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        """
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        """
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        """

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        """
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        """
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        """
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        """
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(
                    ("Invalid placement. row should be in the range 1 to {}.").format(
                        len(board) - 1
                    )
                )
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(
                    (
                        "Invalid placement. column should be in the range 1 to {}."
                    ).format(len(board) - 1)
                )
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print("Invalid placement. There is already a chess in this position.")
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print("Invalid placement. No liberty found in this position.")
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(
                self.previous_board, test_go.board
            ):
                if verbose:
                    print(
                        "Invalid placement. A repeat move not permitted by the KO rule."
                    )
                return False
        return True

    def update_board(self, new_board):
        """
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        """
        self.board = new_board

    def visualize_board(self):
        """
        Visualize the board.

        :return: None
        """
        board = self.board

        print("-" * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(" ", end=" ")
                elif board[i][j] == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()
        print("-" * len(board) * 2)

    def game_end(self, piece_type, action="MOVE", test=False):
        """
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        """

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            if not test:
                # print("GAME END: Case 1: max move reached", self.n_move)
                pass
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            if not test:
                # print("GAME END: Case 2: two players all pass the move.")
                pass
            return True
        return False

    def get_scores(self):
        score1 = 0
        score2 = self.komi

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    score1 += 1
                elif self.board[i][j] == 2:
                    score2 += 1

        return score1, score2

    def judge_winner(self):
        """
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        """
        cnt_1, cnt_2 = self.get_scores()

        if cnt_1 > cnt_2:
            return 1
        elif cnt_1 < cnt_2:
            return 2
        else:
            return 0

    def play(self, player1, player2, verbose=False):
        """
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        """

        self.init_board(self.size)
        # Print input hints and error message if there is a manual player
        if player1.type == "manual" or player2.type == "manual":
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print("X stands for black chess, O stands for white chess.")
            self.visualize_board()

        verbose = self.verbose
        # Game starts!
        while 1:
            piece_type = 1 if self.X_move else 2

            # Judge if the game should end
            if self.game_end(piece_type):
                result = self.judge_winner()
                if verbose:
                    print("Game ended.")
                    if result == 0:
                        print("The game is a tie.")
                    else:
                        print("The winner is {}".format("X" if result == 1 else "O"))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            # Game continues
            if piece_type == 1:
                action = player1.get_input(self, piece_type)
            else:
                action = player2.get_input(self, piece_type)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)
            print(action)
            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board()
                    continue

                self.died_pieces = self.remove_died_pieces(
                    3 - piece_type
                )  # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board()  # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move  # Players take turn


def judge(moves, piece_type, previous_board, board, action, x, y, verbose):
    go = GO(5)
    go.set_board(piece_type, previous_board, board)
    go.n_move = moves

    if action == "MOVE":
        if not go.place_chess(x, y, piece_type):
            return 5, go.board, go.previous_board

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    if verbose:
        go.visualize_board()
        print()

    if go.game_end(piece_type, action):
        result = go.judge_winner()
        # print("Game end.", result)
        return result, go.board, go.previous_board

    if action == "PASS":
        go.previous_board = deepcopy(board)

    return 4, go.board, go.previous_board


def train_agent(agent, total_games, model_name):
    print(f"Training agent for {total_games} games...")

    for game in range(0, total_games + 1):
        exploit = False

        if game % 5000 == 0:
            print(f"Game {game} of {total_games}...", agent.epsilon)
            exploit = True

        moves = 0
        piece_type = 1
        board = np.zeros((5, 5), dtype=np.int8)
        previous_board = np.zeros((5, 5), dtype=np.int8)
        is_game_end = False

        while True:
            transition = agent.get_input(
                moves, previous_board, board, piece_type, exploit, game > 2 * 10**4
            )
            state = np.concatenate(
                (
                    np.array([piece_type % 2, moves, int(is_game_end)]),
                    np.array(previous_board).flatten(),
                    np.array(board).flatten(),
                )
            )

            r, c = transition // 5, transition % 5
            moves += 1

            if transition != 25:
                action = "MOVE"
            else:
                action = "PASS"

            result, board, previous_board = judge(
                moves, piece_type, previous_board, board, action, r, c, False
            )

            if game % 5000 == 0:
                if piece_type % 2:
                    print("Black makes move...", action, r, c, result)
                else:
                    print("White makes move...", action, r, c, result)

                for row in board:
                    print("".join(map(str, row)))
                # time.sleep(1)

            is_game_end = True

            if result == 5:
                reward = -2
            elif result == 4:
                reward = 0.02
                is_game_end = False
            elif result == piece_type:
                reward = 1
            elif result == 0:
                reward = 0.5
            else:
                reward = -1

            next_state = np.concatenate(
                (
                    np.array([piece_type % 2, moves, int(is_game_end)]),
                    np.array(previous_board).flatten(),
                    np.array(board).flatten(),
                )
            )

            agent.buffer.push(Transition(state, transition, reward, next_state))
            piece_type = 3 - piece_type

            if is_game_end:
                break

        if game % 10 == 0:
            agent.learn(1000)

        if game % 1000 == 0:
            agent.save_agent(model_name)

    print("Training phase completed.")


# play function for actual game matches
# def play(black_cmd, white_cmd):
#     print("Clean up...", file=sys.stderr)
#
#     # Clean up old input/output files
#     if os.path.exists("input.txt"):
#         os.remove("input.txt")
#     if os.path.exists("output.txt"):
#         os.remove("output.txt")
#
#     # Copy initial input file
#     os.system(f"cp {prefix}init/input.txt ./input.txt")
#
#     print("Start Playing...", file=sys.stderr)
#
#     moves = 0
#     while True:
#         # Clean up output.txt before each move
#         if os.path.exists("output.txt"):
#             os.remove("output.txt")
#
#         print("Black makes move...", file=sys.stderr)
#         os.system(black_cmd)
#         moves += 1
#
#         rst = os.system(f"python3 {prefix}host.py -m {moves} -v True")
#
#         if rst != 0:
#             break
#
#         if os.path.exists("output.txt"):
#             os.remove("output.txt")
#
#         print("White makes move...", file=sys.stderr)
#         os.system(white_cmd)
#         moves += 1
#
#         rst = os.system(f"python3 {prefix}host.py -m {moves} -v True")
#
#         if rst != 0:
#             break
#
#     return rst


play_time = 10
training_games = 10**7  # Train for 10,000 games

# Start training the agent
print(datetime.now())

# Set up the training process before playing games

go_agent = QLearningAgent()
model_name = "model_cnn.pt"
if os.path.exists(model_name):
    go_agent.load_agent(model_name, train=True)

train_agent(go_agent, training_games, model_name)
go_agent.save_agent(model_name)

# for ta in ta_agent:  # 1 TA player
#     print("")
#     print(f"==Playing with {ta}==")
#     print(datetime.now())
#     ta_cmd = f"python3 {prefix}{ta}{surfix}"
#     black_win_time = 0
#     white_win_time = 0
#     black_tie = 0
#     white_tie = 0
#
#     for round_num in range(1, play_time + 1, 2):
#         # TA takes Black
#         print(f"=====Round {round_num}=====")
#         print("Black:TA White:You")
#         winner = play(ta_cmd, cmd)
#         if winner == 2:
#             print("White(You) win!")
#             white_win_time += 1
#         elif winner == 0:
#             print("Tie.")
#             white_tie += 1
#         else:
#             print("White(You) lose.")
#
#         # Student takes Black
#         print(f"=====Round {round_num + 1}=====")
#         print("Black:You White:TA")
#         winner = play(cmd, ta_cmd)
#         if winner == 1:
#             print("Black(You) win!")
#             black_win_time += 1
#         elif winner == 0:
#             print("Tie.")
#             black_tie += 1
#         else:
#             print("Black(You) lose.")
#
#     print("=====Summary=====")
#     print(
#         f"You play as Black Player | Win: {black_win_time} | Lose: {play_time // 2 - black_win_time - black_tie} | Tie: {black_tie}"
#     )
#     print(
#         f"You play as White Player | Win: {white_win_time} | Lose: {play_time // 2 - white_win_time - black_tie} | Tie: {white_tie}"
#     )
#
# # Clean up remaining files
# for file in ["input.txt", "output.txt"]:
#     if os.path.exists(file):
#         os.remove(file)
#
# print("")
# print("Mission Completed.")
# print(datetime.now())
