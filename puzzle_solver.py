"""
A-Puzzle-A-Day Calendar Puzzle Solver

This solver finds arrangements of 8 puzzle pieces on a calendar board
such that exactly the target month and day remain uncovered.
"""

import streamlit as st
from datetime import date
from typing import List, Tuple, Set, Optional
import numpy as np


# Board layout: 7 columns x 7 rows
# Blocked cells: top-right corner (0,6), (1,6) and bottom-right (6,3-6)
BOARD_ROWS = 7
BOARD_COLS = 7

# Blocked positions (row, col) - not part of the playable board
BLOCKED = {(0, 6), (1, 6), (6, 3), (6, 4), (6, 5), (6, 6)}

# Month positions (row, col) -> month name
MONTHS = {
    (0, 0): "Jan", (0, 1): "Feb", (0, 2): "Mär", (0, 3): "Apr", (0, 4): "Mai", (0, 5): "Jun",
    (1, 0): "Jul", (1, 1): "Aug", (1, 2): "Sep", (1, 3): "Okt", (1, 4): "Nov", (1, 5): "Dez"
}

# Day positions (row, col) -> day number
DAYS = {}
day = 1
for row in range(2, 7):
    for col in range(7):
        if (row, col) not in BLOCKED and day <= 31:
            DAYS[(row, col)] = day
            day += 1

# Month name to position mapping
MONTH_TO_POS = {v: k for k, v in MONTHS.items()}
MONTH_NAMES = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

# Day to position mapping
DAY_TO_POS = {v: k for k, v in DAYS.items()}


def get_all_valid_cells() -> Set[Tuple[int, int]]:
    """Return all valid board positions."""
    cells = set()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (r, c) not in BLOCKED:
                cells.add((r, c))
    return cells


# Define the 8 puzzle pieces as lists of (row, col) offsets from origin
# Traced from the actual puzzle images provided by user
# Total cells: 43 valid - 2 uncovered = 41 cells covered by pieces
# Pieces: 1 hexomino (6 cells) + 7 pentominoes (5 cells) = 6 + 35 = 41 cells

PIECES = [
    # Piece 0: Rectangle-ish hexomino (6 cells) - the teal piece shown outside board
    # X X
    # X X
    # X X
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],

    # Piece 1: L-pentomino (5 cells) - light gray piece top-left
    # X
    # X
    # X
    # X X
    [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],

    # Piece 2: Z/S-pentomino (5 cells)
    # X X
    #   X
    #   X X
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],

    # Piece 3: U-pentomino (5 cells)
    # X   X
    # X X X
    [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],

    # Piece 4: N/S-pentomino variant (5 cells)
    #   X
    # X X
    # X
    # X
    [(0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],

    # Piece 5: P-pentomino (5 cells) - 2x2 block with tail
    # X X
    # X X
    # X
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],

    # Piece 6: Y-pentomino (5 cells)
    #   X
    # X X
    #   X
    #   X
    [(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)],

    # Piece 7: V/corner-pentomino (5 cells)
    # X
    # X
    # X X X
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
]

# Piece colors for visualization - distinct colors for each piece
PIECE_COLORS = [
    "#E74C3C",  # Red
    "#3498DB",  # Blue
    "#2ECC71",  # Green
    "#F39C12",  # Orange
    "#9B59B6",  # Purple
    "#1ABC9C",  # Teal
    "#E91E63",  # Pink
    "#00BCD4",  # Cyan
]


def rotate_piece(piece: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate piece 90 degrees clockwise."""
    # (r, c) -> (c, -r) then normalize
    rotated = [(c, -r) for r, c in piece]
    min_r = min(r for r, c in rotated)
    min_c = min(c for r, c in rotated)
    return [(r - min_r, c - min_c) for r, c in rotated]


def flip_piece(piece: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Flip piece horizontally."""
    flipped = [(r, -c) for r, c in piece]
    min_r = min(r for r, c in flipped)
    min_c = min(c for r, c in flipped)
    return [(r - min_r, c - min_c) for r, c in flipped]


def normalize_piece(piece: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Normalize piece to start at (0,0) and sort cells."""
    min_r = min(r for r, c in piece)
    min_c = min(c for r, c in piece)
    normalized = [(r - min_r, c - min_c) for r, c in piece]
    return tuple(sorted(normalized))


def get_all_orientations(piece: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Get all unique orientations (rotations and flips) of a piece."""
    orientations = set()
    current = piece

    for _ in range(4):  # 4 rotations
        orientations.add(normalize_piece(current))
        orientations.add(normalize_piece(flip_piece(current)))
        current = rotate_piece(current)

    return [list(o) for o in orientations]


def precompute_piece_orientations() -> List[List[List[Tuple[int, int]]]]:
    """Precompute all orientations for all pieces."""
    return [get_all_orientations(piece) for piece in PIECES]


def can_place_piece(
    board: np.ndarray,
    piece: List[Tuple[int, int]],
    start_row: int,
    start_col: int,
    valid_cells: Set[Tuple[int, int]]
) -> bool:
    """Check if piece can be placed at given position."""
    for dr, dc in piece:
        r, c = start_row + dr, start_col + dc
        if (r, c) not in valid_cells:
            return False
        if board[r, c] != -1:
            return False
    return True


def place_piece(
    board: np.ndarray,
    piece: List[Tuple[int, int]],
    start_row: int,
    start_col: int,
    piece_id: int
) -> None:
    """Place piece on board."""
    for dr, dc in piece:
        board[start_row + dr, start_col + dc] = piece_id


def remove_piece(
    board: np.ndarray,
    piece: List[Tuple[int, int]],
    start_row: int,
    start_col: int
) -> None:
    """Remove piece from board."""
    for dr, dc in piece:
        board[start_row + dr, start_col + dc] = -1


def find_first_empty(board: np.ndarray, valid_cells: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Find the first empty cell in reading order."""
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (r, c) in valid_cells and board[r, c] == -1:
                return (r, c)
    return None


def solve(
    board: np.ndarray,
    pieces_orientations: List[List[List[Tuple[int, int]]]],
    used_pieces: List[bool],
    valid_cells: Set[Tuple[int, int]]
) -> bool:
    """Backtracking solver - returns True if solution found."""
    # Find first empty cell
    empty = find_first_empty(board, valid_cells)
    if empty is None:
        return True  # All cells filled

    row, col = empty

    # Try each unused piece
    for piece_id, orientations in enumerate(pieces_orientations):
        if used_pieces[piece_id]:
            continue

        # Try each orientation
        for piece in orientations:
            # Try placing at positions that would cover the empty cell
            for dr, dc in piece:
                start_row = row - dr
                start_col = col - dc

                if can_place_piece(board, piece, start_row, start_col, valid_cells):
                    place_piece(board, piece, start_row, start_col, piece_id)
                    used_pieces[piece_id] = True

                    if solve(board, pieces_orientations, used_pieces, valid_cells):
                        return True

                    remove_piece(board, piece, start_row, start_col)
                    used_pieces[piece_id] = False

    return False


def solve_puzzle(month: int, day: int) -> Optional[np.ndarray]:
    """
    Solve the puzzle for a given month (1-12) and day (1-31).
    Returns the board with piece IDs or None if no solution.
    """
    # Get target positions to leave uncovered
    month_name = MONTH_NAMES[month - 1]
    month_pos = MONTH_TO_POS[month_name]
    day_pos = DAY_TO_POS[day]

    # Set up valid cells (excluding blocked and target date)
    valid_cells = get_all_valid_cells()
    valid_cells.discard(month_pos)
    valid_cells.discard(day_pos)

    # Initialize board: -1 means empty, -2 means blocked/target
    board = np.full((BOARD_ROWS, BOARD_COLS), -2, dtype=int)
    for cell in valid_cells:
        board[cell] = -1

    # Precompute piece orientations
    pieces_orientations = precompute_piece_orientations()
    used_pieces = [False] * len(PIECES)

    if solve(board, pieces_orientations, used_pieces, valid_cells):
        return board
    return None


def render_board(board: Optional[np.ndarray], month: int, day: int) -> str:
    """Render the board as HTML for Streamlit."""
    month_name = MONTH_NAMES[month - 1]

    html = """
    <style>
    .puzzle-board {
        display: inline-block;
        background: #f5e6d3;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .puzzle-row {
        display: flex;
    }
    .puzzle-cell {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        margin: 1px;
        border-radius: 4px;
    }
    .empty-cell {
        background: #f5e6d3;
    }
    .blocked-cell {
        background: transparent;
    }
    .target-cell {
        background: #f5e6d3;
        border: 2px solid #8b7355;
        color: #5a4a3a;
    }
    </style>
    <div class="puzzle-board">
    """

    for r in range(BOARD_ROWS):
        html += '<div class="puzzle-row">'
        for c in range(BOARD_COLS):
            if (r, c) in BLOCKED:
                html += '<div class="puzzle-cell blocked-cell"></div>'
            elif (r, c) in MONTHS:
                cell_month = MONTHS[(r, c)]
                if cell_month == month_name:
                    html += f'<div class="puzzle-cell target-cell">{cell_month}</div>'
                elif board is not None and board[r, c] >= 0:
                    color = PIECE_COLORS[board[r, c]]
                    html += f'<div class="puzzle-cell" style="background:{color};color:#fff;">{cell_month}</div>'
                else:
                    html += f'<div class="puzzle-cell empty-cell">{cell_month}</div>'
            elif (r, c) in DAYS:
                cell_day = DAYS[(r, c)]
                if cell_day == day:
                    html += f'<div class="puzzle-cell target-cell">{cell_day}</div>'
                elif board is not None and board[r, c] >= 0:
                    color = PIECE_COLORS[board[r, c]]
                    html += f'<div class="puzzle-cell" style="background:{color};color:#fff;">{cell_day}</div>'
                else:
                    html += f'<div class="puzzle-cell empty-cell">{cell_day}</div>'
            else:
                html += '<div class="puzzle-cell empty-cell"></div>'
        html += '</div>'

    html += '</div>'
    return html


def main():
    st.set_page_config(
        page_title="A-Puzzle-A-Day Solver",
        page_icon="📅",
        layout="centered"
    )

    st.title("📅 A-Puzzle-A-Day Solver")
    st.markdown("""
    This solver finds the arrangement of puzzle pieces that leaves exactly
    your chosen date (month and day) visible on the calendar board.
    """)

    # Date selection
    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: MONTH_NAMES[x - 1],
            index=date.today().month - 1
        )

    with col2:
        max_day = 31
        if month in [4, 6, 9, 11]:
            max_day = 30
        elif month == 2:
            max_day = 29

        current_day = min(date.today().day, max_day)
        day_val = st.selectbox(
            "Day",
            options=list(range(1, max_day + 1)),
            index=current_day - 1
        )

    if st.button("🧩 Solve Puzzle", type="primary", use_container_width=True):
        with st.spinner("Finding solution..."):
            solution = solve_puzzle(month, day_val)

        if solution is not None:
            st.success(f"Solution found for {MONTH_NAMES[month - 1]} {day_val}!")
            st.markdown(render_board(solution, month, day_val), unsafe_allow_html=True)
        else:
            st.error("No solution found. This shouldn't happen - please check the puzzle configuration.")

    # Show empty board on initial load
    else:
        st.markdown("### Preview")
        st.markdown(render_board(None, month, day_val), unsafe_allow_html=True)
        st.caption("Click 'Solve Puzzle' to find the solution")


if __name__ == "__main__":
    main()
