import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    utilities = [row_strategy @ row_matrix @ col_strategy,
                 row_strategy @ col_matrix @ col_strategy]
    return np.array(utilities)


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    utility = row_strategy @ row_matrix @ col_strategy
    return np.array([utility, -utility])


def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    payoff = row_strategy @ col_matrix
    strategy = np.zeros(col_matrix.shape[1])
    strategy[np.argmax(payoff)] = 1.0
    return strategy


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    payoff = row_matrix @ col_strategy
    strategy = np.zeros(row_matrix.shape[0])
    strategy[np.argmax(payoff)] = 1.0
    return strategy


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float64:
    col_strategy = calculate_best_response_against_row(col_matrix, row_strategy)
    return row_strategy @ row_matrix @ col_strategy


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    row_strategy = calculate_best_response_against_col(row_matrix, col_strategy)
    return row_strategy @ col_matrix @ col_strategy


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    dominated = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j and np.all(matrix[j] > matrix[i]):
                dominated.append(i)
                break
    return np.array(dominated, dtype=np.int64)


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_matrix = np.array(row_matrix, dtype=np.float64)
    col_matrix = np.array(col_matrix, dtype=np.float64)
    row_actions = np.arange(row_matrix.shape[0], dtype=np.int64)
    col_actions = np.arange(row_matrix.shape[1], dtype=np.int64)

    while True:
        dominated_rows = find_strictly_dominated_actions(row_matrix)
        dominated_cols = find_strictly_dominated_actions(col_matrix.T)
        if len(dominated_rows) == 0 and len(dominated_cols) == 0:
            break

        keep_rows = np.ones(len(row_actions), dtype=bool)
        keep_cols = np.ones(len(col_actions), dtype=bool)
        keep_rows[dominated_rows] = False
        keep_cols[dominated_cols] = False
        remaining = np.ix_(keep_rows, keep_cols)
        row_matrix = row_matrix[remaining]
        col_matrix = col_matrix[remaining]
        row_actions = row_actions[keep_rows]
        col_actions = col_actions[keep_cols]

    return row_matrix, col_matrix, row_actions, col_actions


def main() -> None:
    row_matrix = np.array([[13, 1, 7], [4, 3, 6], [-1, 2, 8]])
    col_matrix = np.array([[3, 4, 3], [1, 3, 2], [9, 8, -1]])
    for item in iterated_removal_of_dominated_strategies(row_matrix, col_matrix):
        print(item)


if __name__ == '__main__':
    main()
