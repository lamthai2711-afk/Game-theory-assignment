from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    if row_matrix.shape[0] != 2:
        raise ValueError('Expected a 2xN game')

    x = np.append(np.arange(0, 1, step_size), 1)
    strategies = np.column_stack((x, 1 - x))
    values = strategies @ row_matrix
    plt.plot(x, values.min(axis=1))
    plt.xlabel('Probability of first action')
    plt.ylabel('Best response value')
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    payoffs = matrix[np.ix_(row_support, col_support)]
    n = len(col_support)
    equations = np.column_stack((payoffs, -np.ones(len(row_support))))
    probability_sum = np.append(np.ones(n), 0.0)
    right_side = np.append(np.zeros(len(row_support)), 1.0)

    result = linprog(
        np.zeros(n + 1),
        A_eq=np.vstack((equations, probability_sum)),
        b_eq=right_side,
        bounds=[(0, None)] * n + [(None, None)],
    )
    if not result.success:
        return None
    return result.x[:n]


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    row_actions = range(row_matrix.shape[0])
    col_actions = range(row_matrix.shape[1])
    row_supports = []
    col_supports = []
    for size in range(1, row_matrix.shape[0] + 1):
        row_supports.extend(combinations(row_actions, size))
    for size in range(1, row_matrix.shape[1] + 1):
        col_supports.extend(combinations(col_actions, size))

    equilibria = []
    for r, c in product(row_supports, col_supports):
        row_support = np.array(r)
        col_support = np.array(c)
        s = verify_support(row_matrix, row_support, col_support)
        p = verify_support(col_matrix.T, col_support, row_support)
        if p is None or s is None:
            continue

        row_strategy = np.zeros(row_matrix.shape[0])
        col_strategy = np.zeros(row_matrix.shape[1])
        row_strategy[row_support] = p
        col_strategy[col_support] = s

        row_payoffs = row_matrix @ col_strategy
        col_payoffs = row_strategy @ col_matrix
        if np.max(row_payoffs[row_support]) < np.max(row_payoffs) - 1e-8:
            continue
        if np.max(col_payoffs[col_support]) < np.max(col_payoffs) - 1e-8:
            continue

        pair = row_strategy, col_strategy
        if not any(all(np.array_equal(a, b) for a, b in zip(pair, old))
                   for old in equilibria):
            equilibria.append(pair)

    return equilibria


def main() -> None:
    row_matrix = np.array([[0, 0, -10], [1, -10, -10], [-10, -10, -10]])
    col_matrix = np.array([[0, 1, -10], [0, -10, -10], [-10, -10, -10]])
    for row_strategy, col_strategy in support_enumeration(row_matrix, col_matrix):
        print(row_strategy, col_strategy)
    game = np.array([[-1, 0, -0.8], [1, -1, -0.5]])
    plot_best_response_value_function(game, 0.01)


if __name__ == '__main__':
    main()
