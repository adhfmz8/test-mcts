import random
import math
import sys
import argparse
import time
import copy
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set

# --- Constants ---
ROWS: int = 6
COLS: int = 7
EMPTY: str = "O"
RED: str = "R"  # Min player (-1)
YELLOW: str = "Y"  # Max player (+1)
PLAYERS: Set[str] = {RED, YELLOW}
PLAYER_VALS: Dict[Optional[str], int] = {
    RED: -1,
    YELLOW: 1,
    None: 0,
    "D": 0,
}  # Added "D" explicitly

# --- Precomputed Values ---
SQRT_2 = math.sqrt(2)  # Precompute for UCB default

# --- Game Logic Class ---


class ConnectFour:
    """Represents the Connect Four game state and rules."""

    # Add type hints for attributes
    board: List[List[str]]
    current_player: Optional[str]
    last_move: Optional[Tuple[int, int]]

    def __init__(
        self,
        board: Optional[List[List[str]]] = None,
        current_player: Optional[str] = None,
    ):
        if board:
            if len(board) != ROWS or any(len(r) != COLS for r in board):
                raise ValueError("Invalid board dimensions provided.")
            self.board = [list(row) for row in board]  # Ensure mutable lists
        else:
            self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]

        self.current_player = current_player
        self.last_move = None  # Store (row, col) of the last move

    def get_legal_moves(self) -> List[int]:
        """Returns a list of column indices (0-6) where a piece can be dropped."""
        # This is already efficient O(COLS)
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def is_column_full(self, col: int) -> bool:
        """Checks if a column is full."""
        return self.board[0][col] != EMPTY

    def make_move(self, col: int, player: str) -> Optional[int]:
        """
        Places a piece for the player in the specified column.
        Returns the row where the piece landed, or None if illegal.
        Updates self.last_move. Optimized to avoid re-checking legality if possible.
        """
        # Check column bounds first (slightly faster than checking full if often illegal)
        if not (0 <= col < COLS):
            return None  # Illegal move: out of bounds

        # Check if full (direct access, fast)
        if self.board[0][col] != EMPTY:
            return None  # Illegal move: column full

        # Find empty slot - iterates ROWS in worst case, but typically faster
        # Starts from bottom, which is common access pattern
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = player
                self.last_move = (r, col)
                return r
        # Should be unreachable if is_column_full works correctly, but acts as safeguard
        return None  # Should not happen

    def undo_specific_move(self, row: int, col: int) -> bool:
        """Undoes a move at a specific cell. Essential if implementing undo-based MCTS."""
        # This remains O(1). Critical for the more advanced undo-based optimization path
        # (which is not the primary path taken in the current MCTS implementation).
        if 0 <= row < ROWS and 0 <= col < COLS and self.board[row][col] != EMPTY:
            self.board[row][col] = EMPTY

            return True
        return False

    def check_win(self, player: str) -> bool:
        """
        Checks if the given player has won. Optimized to check around last move.
        Returns False if no move has been made yet.
        """
        if not self.last_move:
            return False  # Cannot win if no move has been made

        r_last, c_last = self.last_move
        # Ensure the piece at last_move actually belongs to the player being checked
        # (Sanity check, should be true if called correctly after make_move)
        if self.board[r_last][c_last] != player:
            return False  # Should not happen normally

        # Directions: Horizontal, Vertical, Diagonal Up-Right, Diagonal Down-Right
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Start with the piece at last_move

            # Check positive direction
            for i in range(1, 4):
                r, c = r_last + i * dr, c_last + i * dc
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                    count += 1
                else:
                    break  # Out of bounds or different piece

            # Check negative direction
            for i in range(1, 4):
                r, c = r_last - i * dr, c_last - i * dc
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                    count += 1
                else:
                    break  # Out of bounds or different piece

            if count >= 4:
                return True  # Found a win

        return False  # No win found centered on last_move

    def check_draw(self) -> bool:
        """Checks if the game is a draw (board full)."""
        # Checking the top row is sufficient and efficient O(COLS)
        # return all(self.board[0][c] != EMPTY for c in range(COLS))
        # Or reuse get_legal_moves (slightly less direct but conceptually clear)
        return len(self.get_legal_moves()) == 0

    def get_winner(self) -> Optional[str]:
        """
        Determines the winner or if it's a draw.
        Returns player ('R' or 'Y'), 'D' for draw, or None if ongoing.
        Optimized: Checks win only for the player who could have just won.
        """
        if not self.last_move:  # No moves made yet
            return None

        # Player who made the last move is the only one who could have won
        # This assumes last_move is correctly set by make_move
        last_player = self.board[self.last_move[0]][self.last_move[1]]
        if self.check_win(last_player):
            return last_player

        # Check for draw only if no winner
        if self.check_draw():
            return "D"

        return None  # Game is ongoing

    def get_outcome_value(self) -> int:
        """Returns the numerical value of the current state (-1, 0, 1)."""
        # This function isn't used directly by MCTS simulation logic which uses the winner 'R','Y','D'
        # It's more for evaluating a terminal state if needed elsewhere.
        winner = self.get_winner()
        return PLAYER_VALS.get(winner, 0)  # Use PLAYER_VALS mapping

    @staticmethod  # Static method as it doesn't depend on instance state
    def get_opponent(player: Optional[str]) -> Optional[str]:
        """Gets the opponent of the given player."""
        if player == RED:
            return YELLOW
        elif player == YELLOW:
            return RED
        else:
            return None

    def print_board(self, file=sys.stdout) -> None:
        """Prints the board to the console or a file."""
        # Keep as is, used for display, not performance critical path.
        print("  ".join(map(str, range(1, COLS + 1))), file=file)
        print("-" * (COLS * 3 - 1), file=file)
        for row in self.board:
            print("  ".join(row), file=file)
        print("", file=file)

    def get_board_string(self) -> str:
        """Returns the board state as a multi-line string."""
        # Keep as is, used potentially for hashing/debugging.
        return "\n".join("".join(row) for row in self.board)


# --- MCTS Node ---


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    parent: Optional["MCTSNode"]
    move: Optional[int]  # Column index that led to this node
    player_to_move: Optional[str]  # Player whose turn it is at this node's state
    wins: float  # Sum of outcomes from simulations (-1, 0, 1) from YELLOW's perspective
    visits: int
    children: Dict[int, "MCTSNode"]  # move -> child_node
    untried_moves: List[int]

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        move: Optional[int] = None,
        player_to_move: Optional[str] = None,
        untried_moves: Optional[List[int]] = None,
    ):
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.wins = 0.0
        self.visits = 0
        self.children = {}
        self.untried_moves = untried_moves if untried_moves is not None else []
        # Shuffling here ensures random expansion order, good for exploration
        random.shuffle(self.untried_moves)

    def is_fully_expanded(self) -> bool:
        """Checks if all legal moves from this node have been explored."""
        return not self.untried_moves  # More Pythonic check for empty list

    def add_child(
        self, move: int, player_to_move: str, untried_moves_for_child: List[int]
    ) -> "MCTSNode":
        """Adds a new child node for the given move. Assumes move is valid and untried."""
        # Optimization: Direct creation, assumes 'move' was popped from untried_moves before calling
        node = MCTSNode(
            parent=self,
            move=move,
            player_to_move=player_to_move,
            untried_moves=untried_moves_for_child,
        )
        self.children[move] = node
        return node

    def update(self, result: int) -> None:
        """Updates the node's statistics after a simulation."""
        # Direct updates are fast.
        self.visits += 1
        self.wins += (
            result  # Result is always from Yellow's perspective (+1 Y win, -1 R win)
        )

    def ucb_score(self, C: float = SQRT_2) -> float:
        """
        Calculates the UCB1 score for this node.
        Uses the precomputed SQRT_2 by default.
        """
        # Optimization: Handle visits == 0 directly
        if self.visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        if not self.parent:  # Should not happen for children, maybe root if called?
            return 0.0  # Or some other default for root?

        # Value term (exploitation): Average win rate *from the perspective of the parent*
        # Node's wins are stored from Yellow's perspective.
        # Parent wants to maximize its own outcome.
        average_win_rate = self.wins / self.visits
        if self.parent.player_to_move == RED:  # If parent is MIN player
            exploitation_term = (
                -average_win_rate
            )  # MIN wants to minimize Yellow's score (maximize Red's)
        else:  # Parent is MAX player (YELLOW)
            exploitation_term = average_win_rate  # MAX wants to maximize Yellow's score

        # Exploration term
        # Optimization: Use cached log if parent visits doesn't change often (not easy here)
        exploration_term = C * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation_term + exploration_term

    def __repr__(self) -> str:
        # Keep for debugging
        val = self.wins / self.visits if self.visits > 0 else 0
        move_repr = self.move + 1 if self.move is not None else "Root"
        return (
            f"[M={move_repr} P={self.player_to_move} W/N={self.wins:.1f}/{self.visits}"
            f" Val={val:.2f} U={len(self.untried_moves)} C={len(self.children)}]"
        )


# --- Algorithms ---


def select_move_ur(game: ConnectFour) -> Optional[int]:
    """Selects a move using the Uniform Random strategy."""
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def run_mcts(
    game_state: ConnectFour,
    player: str,
    num_simulations: int,
    algorithm_type: str,
    verbosity: str,
    C: float = SQRT_2,
) -> Optional[int]:
    """
    Runs Monte Carlo Tree Search (PMCGS or UCT) to select the best move.

    Args:
        game_state: The current state of the game.
        player: The player ('R' or 'Y') whose turn it is.
        num_simulations: The number of simulations to run.
        algorithm_type: 'PMCGS' or 'UCT'.
        verbosity: 'Verbose', 'Brief', or 'None'.
        C: Exploration constant for UCT.

    Returns:
        The selected column index (0-6), or None if no moves possible.
    """
    start_time = time.time()

    # Create root node representing the current actual game state
    initial_legal_moves = game_state.get_legal_moves()
    if not initial_legal_moves:
        print(
            "Warning: MCTS called on terminal state or no moves possible.",
            file=sys.stderr,
        )
        return None  # No moves to select

    root = MCTSNode(player_to_move=player, untried_moves=initial_legal_moves)

    # root_game = game_state.copy() # If copy method exists
    root_game_board = [list(r) for r in game_state.board]
    root_game_last_move = game_state.last_move

    # Precompute opponent for the root player
    root_opponent = ConnectFour.get_opponent(player)

    for i in range(num_simulations):
        if verbosity == "Verbose":
            print(f"\n--- Simulation {i+1} ---")

        node = root

        sim_game = ConnectFour(
            board=[list(r) for r in root_game_board], current_player=player
        )
        sim_game.last_move = root_game_last_move

        visited_nodes_path = [node]
        winner = None  # Track if game ends during selection/expansion

        # --- 1. Selection ---
        while node.is_fully_expanded() and node.children:
            # Node is internal and fully expanded, must select a child
            if verbosity == "Verbose":
                print(f"Selecting UCT/PMCGS from Node: {node}")

            if algorithm_type == "UCT":
                # Find child with the highest UCB score.
                # Using max with a key function is concise.
                # Shuffling items beforehand ensures random tie-breaking.
                child_items = list(node.children.items())
                random.shuffle(child_items)
                selected_move, node = max(
                    child_items, key=lambda item: item[1].ucb_score(C)
                )

                # Debug print UCB scores (optional, controlled by verbosity)
                if verbosity == "Verbose":
                    for m_idx, ch in sorted(
                        node.parent.children.items()
                    ):  # Access parent's children
                        print(
                            f" V{m_idx+1}: {ch.ucb_score(C):.3f} (W/N:{ch.wins:.1f}/{ch.visits})",
                            end="",
                        )
                    print(f" -> Chose: {selected_move + 1}")

            else:  # PMCGS - select randomly among existing children
                selected_move = random.choice(list(node.children.keys()))
                node = node.children[selected_move]
                if verbosity == "Verbose":
                    print(f" -> Chose Random: {selected_move + 1}")

            # Make the selected move on the simulation board
            sim_game.make_move(
                selected_move, sim_game.current_player
            )  # Player was parent's player
            sim_game.current_player = ConnectFour.get_opponent(sim_game.current_player)
            visited_nodes_path.append(node)

            winner = sim_game.get_winner()
            if winner is not None:
                break  # Game ended during selection traversal

        # --- 2. Expansion ---
        # If we reached a node that isn't terminal and isn't fully expanded
        if winner is None and not node.is_fully_expanded():
            move = node.untried_moves.pop()  # Random due to initial shuffle
            if verbosity == "Verbose":
                print(f"Expanding move {move+1} from Node: {node}")

            player_making_move = sim_game.current_player
            sim_game.make_move(move, player_making_move)
            opponent = ConnectFour.get_opponent(player_making_move)
            sim_game.current_player = opponent  # Turn switches

            # Determine legal moves for the new state
            winner = sim_game.get_winner()  # Check if expansion ended the game
            child_legal_moves = []
            if winner is None:
                child_legal_moves = sim_game.get_legal_moves()

            # Add the new child node
            node = node.add_child(move, opponent, child_legal_moves)
            visited_nodes_path.append(node)
            if verbosity == "Verbose":
                print(f" Added Child Node: {node}")

        # --- 3. Simulation (Rollout) ---
        # Simulate from the new node (or the terminal node reached) until the game ends
        rollout_start_player = sim_game.current_player
        rollout_last_move = sim_game.last_move  # Track for win checks

        if verbosity == "Verbose":
            print("Starting Rollout...")
        rollout_path_debug = []  # Only for verbose printing

        while winner is None:
            possible_moves = sim_game.get_legal_moves()
            if not possible_moves:
                winner = "D"  # Draw if no moves left
                break

            rollout_move = random.choice(possible_moves)
            if verbosity == "Verbose":
                rollout_path_debug.append(rollout_move + 1)

            # Make the random move
            sim_game.make_move(rollout_move, sim_game.current_player)
            winner = sim_game.get_winner()  # Check win/draw immediately
            if winner is None:
                sim_game.current_player = ConnectFour.get_opponent(
                    sim_game.current_player
                )

        if verbosity == "Verbose":
            print(f" Rollout Path: {rollout_path_debug} -> Winner: {winner}")

        # --- 4. Backpropagation ---
        result = PLAYER_VALS.get(
            winner, 0
        )  # Get numerical result (-1 R win, 0 Draw, 1 Y win)
        if verbosity == "Verbose":
            print(f"Backpropagating Result: {result}")

        # Update stats for all nodes visited in this simulation
        for visited_node in reversed(visited_nodes_path):
            visited_node.update(result)
            if verbosity == "Verbose":
                print(f" Updated Node: {visited_node}")

    # --- Select Final Move ---
    # After all simulations, choose the move from the root with the best score.
    # Typically based on robustness (most visits) or highest win rate.
    # Using highest average value (win rate) is common.
    if not root.children:
        # No simulations run, or immediate end state? Fallback.
        # This case was handled earlier by checking initial_legal_moves
        # If root has no children after sims, means sims never expanded? (e.g., num_simulations=0)
        # Or the game ended immediately upon first simulated move?
        if initial_legal_moves:
            print(
                "Warning: No children explored in MCTS, choosing random.",
                file=sys.stderr,
            )
            return random.choice(initial_legal_moves)
        else:
            print("Error: No legal moves and no children after MCTS.", file=sys.stderr)
            return None  # Should have been caught earlier

    best_final_move = -1
    # Select based on highest win rate (for Yellow) or lowest (for Red)
    # Note: Node values (wins) are stored relative to Yellow.
    if player == YELLOW:  # Max player wants highest win rate
        best_avg_value = -float("inf")
        # Choose move leading to child with highest win_rate (wins/visits)
        # Filter out children with 0 visits if any (shouldn't happen if sims > 0)
        valid_children = {m: c for m, c in root.children.items() if c.visits > 0}
        if not valid_children:  # Fallback if no children were visited
            return random.choice(initial_legal_moves)  # Or handle differently

        best_final_move = max(
            valid_children,
            key=lambda m: valid_children[m].wins / valid_children[m].visits,
        )
        best_avg_value = (
            valid_children[best_final_move].wins
            / valid_children[best_final_move].visits
        )

    else:  # player == RED (Min player wants lowest win rate for Yellow)
        best_avg_value = float("inf")
        # Choose move leading to child with lowest win_rate (wins/visits)
        valid_children = {m: c for m, c in root.children.items() if c.visits > 0}
        if not valid_children:
            return random.choice(initial_legal_moves)

        best_final_move = min(
            valid_children,
            key=lambda m: valid_children[m].wins / valid_children[m].visits,
        )
        best_avg_value = (
            valid_children[best_final_move].wins
            / valid_children[best_final_move].visits
        )

    # --- Print Final Output ---
    if verbosity in ["Brief", "Verbose"]:
        print("\n--- Final Evaluations (Win Rate for Yellow) ---")
        child_vals = []
        for col in range(COLS):
            child = root.children.get(col)
            if child and child.visits > 0:
                win_rate = child.wins / child.visits
                child_vals.append(
                    f"Col {col+1}: {win_rate:+.3f} ({child.visits} visits)"
                )
            elif (
                col in initial_legal_moves
            ):  # Was legal but maybe never explored/visited
                child_vals.append(f"Col {col+1}: --- (0 visits)")
            else:  # Column was full initially
                child_vals.append(f"Col {col+1}: [Full]")
        print(" ".join(child_vals))

    # Ensure a valid move is returned, fallback if selection failed unexpectedly
    if best_final_move == -1:
        if initial_legal_moves:
            print(
                "Warning: MCTS best move selection failed, choosing random.",
                file=sys.stderr,
            )
            best_final_move = random.choice(initial_legal_moves)
        else:  # Game must have been over initially
            return None

    if verbosity != "None":
        print(
            f"FINAL Move selected: {best_final_move + 1}"
        )  # User-friendly 1-based index

    end_time = time.time()
    if verbosity != "None":
        print(f"MCTS took {end_time - start_time:.3f} seconds.")

    return best_final_move


# --- File Parsing ---
def parse_input_file(filepath: str) -> Tuple[str, str, List[List[str]]]:
    """Parses the input file to get algorithm, player, and board state."""
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2 + ROWS:
            raise ValueError("Input file has incorrect number of lines.")

        algorithm = lines[0].upper()
        player = lines[1].upper()
        board_lines = lines[2 : 2 + ROWS]

        if algorithm not in ["UR", "PMCGS", "UCT"]:
            raise ValueError(f"Invalid algorithm specified: {algorithm}")
        if player not in PLAYERS:
            raise ValueError(f"Invalid player specified: {player}")
        if len(board_lines) != ROWS or any(len(row) != COLS for row in board_lines):
            raise ValueError("Invalid board dimensions in file.")

        valid_chars = {RED, YELLOW, EMPTY}
        board = []
        for r_idx, row_str in enumerate(board_lines):
            if any(char not in valid_chars for char in row_str):
                raise ValueError(f"Invalid characters found on board row {r_idx}.")
            board.append(list(row_str))

        return algorithm, player, board

    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred reading the file: {e}", file=sys.stderr)
        sys.exit(1)


# --- Part II: Tournament ---


def play_game(
    player1_info: Tuple[str, int],
    player2_info: Tuple[str, int],
    verbosity: str = "None",
) -> str:
    """
    Plays a full game between two algorithms. Player 1 is Yellow, Player 2 is Red.

    Args:
        player1_info: (algo_name_str, param_int) for player 1 (Yellow)
        player2_info: (algo_name_str, param_int) for player 2 (Red)
        verbosity: Verbosity level for MCTS calls ('None' recommended for speed).

    Returns:
        Winner ('Y', 'R', or 'D' for draw).
    """
    game = ConnectFour()  # Start with empty board
    players_map = {YELLOW: player1_info, RED: player2_info}
    current_player = YELLOW  # Yellow always starts

    move_count = 0
    max_moves = ROWS * COLS

    while move_count < max_moves:
        algo, param = players_map[current_player]
        move = -1

        if algo == "UR":
            move = select_move_ur(game)
        elif algo in ["PMCGS", "UCT"]:
            # Ensure param is positive for MCTS variants
            mcts_param = (
                max(1, param) if param == 0 else param
            )  # Use at least 1 sim if param=0 given? Or error? Let's use 1.
            move = run_mcts(game, current_player, mcts_param, algo, verbosity, C=SQRT_2)
        else:
            print(
                f"Internal Error: Unknown algorithm '{algo}' in play_game.",
                file=sys.stderr,
            )
            return (
                ConnectFour.get_opponent(current_player) or "D"
            )  # Assign loss to current player

        # Validate move
        if move is None or move < 0 or move >= COLS or game.is_column_full(move):
            print(
                f"Error: Algorithm {algo}({param}) for player {current_player} returned invalid move {move+1}. Board:",
                file=sys.stderr,
            )
            game.print_board(file=sys.stderr)
            # Assign loss to the player that made an illegal move
            return ConnectFour.get_opponent(current_player) or "D"

        # Make the move
        row = game.make_move(move, current_player)
        # This check should be redundant due to validation above, but as a safeguard:
        if row is None:
            print(
                f"Internal Error: make_move failed for validated move {move+1}. Board:",
                file=sys.stderr,
            )
            game.print_board(file=sys.stderr)
            return ConnectFour.get_opponent(current_player) or "D"

        # Check for winner
        winner = game.get_winner()
        if winner:
            # print(f"Game End: Winner {winner}") # Debug
            return winner  # 'Y', 'R', or 'D'

        # Switch player
        current_player = ConnectFour.get_opponent(current_player)
        move_count += 1

    # Should be caught by game.get_winner() returning 'D' earlier, but safeguard for max moves
    return "D"


def run_tournament(algorithms: Dict[str, Tuple[str, int]], num_games: int = 100):
    """
    Runs a round-robin tournament and prints results.
    Args:
        algorithms: Dict mapping unique label (e.g., "UCT(500)") to tuple (base_algo_name, param).
        num_games: Number of games per matchup pairing (e.g., P1 vs P2).
    """
    print(f"\n--- Running Tournament ({num_games} games per matchup) ---")
    # Store results: wins[player_label][opponent_label] = win_count
    results = defaultdict(lambda: defaultdict(int))
    draws = defaultdict(lambda: defaultdict(int))
    # Convert dict to list for ordered iteration
    algo_list = list(algorithms.items())  # List of (label, (name, param)) tuples

    for i in range(len(algo_list)):
        for j in range(len(algo_list)):
            if i == j:
                continue  # Skip playing against self

            # Correctly unpack: label, and the tuple (base_name, parameter)
            p1_label, (p1_name, p1_param) = algo_list[i]
            p2_label, (p2_name, p2_param) = algo_list[j]

            print(f"Playing: {p1_label} (Yellow) vs {p2_label} (Red)...")
            p1_wins_count = 0
            draw_count = 0

            # Play N games with P1 as Yellow (player1), P2 as Red (player2)
            for game_num in range(num_games):
                # Pass the base name and param correctly
                winner = play_game(
                    (p1_name, p1_param), (p2_name, p2_param), verbosity="None"
                )
                if winner == YELLOW:  # Player 1 (Yellow) won
                    p1_wins_count += 1
                elif winner == "D":
                    draw_count += 1
                # else: Player 2 (Red) won, no counter needed here for p1 wins

            # Store results for P1 vs P2
            results[p1_label][p2_label] = p1_wins_count
            draws[p1_label][p2_label] = draw_count
            print(
                f"  Result: {p1_label} wins: {p1_wins_count}/{num_games}, Draws: {draw_count}/{num_games}"
            )

    # --- Print Results Table ---
    print("\n--- Tournament Results (Win % for Row Player vs Column Player) ---")
    # Get labels in the order they were processed
    labels = [label for label, _ in algo_list]
    col_width = 16  # Adjust width as needed based on label length
    header = f"{' ':<{col_width}}" + "".join(
        [f"{col_label:>{col_width}}" for col_label in labels]
    )
    print(header)
    print("-" * len(header))

    for row_label in labels:
        row_str = f"{row_label:<{col_width}}"
        for col_label in labels:
            if row_label == col_label:
                row_str += f"{'---':>{col_width}}"
            else:
                # Games played where row_label was Yellow and col_label was Red
                wins = results[row_label].get(col_label, 0)
                draws_count = draws[row_label].get(col_label, 0)
                # Losses = num_games - wins - draws_count

                # Check if the matchup was actually played (it should have been if i != j)
                if num_games > 0:
                    # Calculate win percentage for the row player against the column player
                    win_pct = (wins / num_games) * 100
                    win_pct_str = f"{win_pct:.1f}%"
                elif num_games == 0:
                    win_pct_str = "N/A"
                else:  # Should not happen if check above works
                    win_pct_str = "ERR"

                row_str += f"{win_pct_str:>{col_width}}"
        print(row_str)


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four AI using MCTS variants.")
    parser.add_argument(
        "input_file", help="Path to input file (Part I) or ignored if --run_tournament."
    )
    parser.add_argument(
        "verbosity",
        choices=["Verbose", "Brief", "None"],
        help="Level of output detail.",
    )
    parser.add_argument(
        "parameter", type=int, help="Simulations (MCTS) or 0 (UR). Used for Part I."
    )
    parser.add_argument(
        "--run_tournament", action="store_true", help="Run Part II tournament."
    )
    parser.add_argument(
        "--tournament_games",
        type=int,
        default=50,
        help="Games per matchup in tournament.",
    )  # Reduced default for faster testing

    args = parser.parse_args()

    if args.run_tournament:
        # --- Part II Execution ---
        # Define tournament configurations using unique labels
        tournament_configs = [
            ("UR", 0),
            ("PMCGS", 500),
            ("PMCGS", 5000),
            ("UCT", 500),
            ("UCT", 5000),
        ]
        # Create the dictionary expected by run_tournament {label: (name, param)}
        tournament_dict = {
            f"{name}({param})": (name, param) for name, param in tournament_configs
        }

        run_tournament(tournament_dict, num_games=args.tournament_games)

    else:
        # --- Part I Execution ---
        algo_from_file, player_to_move, initial_board = parse_input_file(
            args.input_file
        )

        # Use settings from command line arguments primarily for consistency
        # Spec implies file dictates algo, but param comes from cmd line. Let's follow that.
        requested_algorithm = algo_from_file
        num_simulations_or_param = args.parameter  # Parameter from command line

        print(f"Algorithm: {requested_algorithm}")
        print(f"Player: {player_to_move}")
        print(f"Parameter/Simulations: {num_simulations_or_param}")
        print("Initial Board:")

        game = ConnectFour(board=initial_board, current_player=player_to_move)

        # --- Heuristic for last_move from initial board (Still Imperfect) ---
        # Goal: Find the coordinates of the last piece placed by the opponent.
        # Method: Count pieces. Determine who moved last. Find the highest piece of that player.
        r_count = sum(row.count(RED) for row in game.board)
        y_count = sum(row.count(YELLOW) for row in game.board)

        last_player_to_move = None
        if (
            player_to_move == YELLOW and y_count == r_count
        ):  # Yellow moves next, counts equal -> Red moved last
            last_player_to_move = RED
        elif (
            player_to_move == RED and y_count > r_count
        ):  # Red moves next, Yellow has more -> Yellow moved last
            last_player_to_move = YELLOW
        # Add cases if board loaded mid-game might not follow strict alternation
        elif y_count > r_count:  # Yellow has more pieces, Yellow likely moved last
            last_player_to_move = YELLOW
        elif r_count > y_count:  # Red has more pieces, Red likely moved last
            last_player_to_move = RED
        elif (
            r_count == y_count and r_count > 0
        ):  # Equal non-zero pieces, Red likely moved last (as Yellow starts)
            last_player_to_move = RED

        if last_player_to_move:
            last_r, last_c = -1, -1
            # Find the highest piece (lowest row index) of the last player
            min_r = ROWS
            found = False
            for r in range(ROWS):
                for c in range(COLS):
                    if game.board[r][c] == last_player_to_move:
                        if r < min_r:
                            # This simple approach just takes the first highest piece found.
                            # A better heuristic might check columns - the highest piece in the
                            # column most recently played? Still complex.
                            # This provides *a* guess for last_move for the optimized check_win.
                            min_r = r
                            last_r, last_c = r, c
                            found = True  # Found at least one piece
            if found:
                game.last_move = (last_r, last_c)
                # print(f"Debug: Guessed last move by {last_player_to_move} at ({last_r}, {last_c})")

        game.print_board()

        # --- Check if game already over ---
        winner = game.get_winner()  # Use the method which relies on last_move if set
        if winner:
            print(f"Game is already over! Result: {winner}")  # Winner or 'D'
        else:
            # --- Run Algorithm ---
            selected_move_col = None
            if requested_algorithm == "UR":
                if num_simulations_or_param != 0:
                    print(
                        "Warning: UR algorithm specified, but parameter is non-zero.",
                        file=sys.stderr,
                    )
                selected_move_col = select_move_ur(game)
                if selected_move_col is not None:
                    # MCTS function prints the final move, UR should too for consistency
                    print(f"FINAL Move selected: {selected_move_col + 1}")
                else:
                    print(
                        "Error: No legal moves available for UR.", file=sys.stderr
                    )  # Should be caught by winner check

            elif requested_algorithm in ["PMCGS", "UCT"]:
                if num_simulations_or_param <= 0:
                    print(
                        f"Warning: {requested_algorithm} requires > 0 simulations, parameter was {num_simulations_or_param}. Running with 1 sim.",
                        file=sys.stderr,
                    )
                    num_simulations_or_param = 1  # Force at least one simulation

                selected_move_col = (
                    run_mcts(  # run_mcts prints the final move internally
                        game,
                        player_to_move,
                        num_simulations_or_param,
                        requested_algorithm,
                        args.verbosity,
                        C=SQRT_2,  # Use default C
                    )
                )
                # Error handling if MCTS returns None (e.g., no legal moves)
                if selected_move_col is None and game.get_legal_moves():
                    print(
                        "Error: MCTS failed to select a move despite available moves.",
                        file=sys.stderr,
                    )

            else:
                # This case should be caught by parse_input_file, but as a safeguard:
                print(
                    f"Error: Unknown algorithm '{requested_algorithm}' specified.",
                    file=sys.stderr,
                )
                sys.exit(1)
