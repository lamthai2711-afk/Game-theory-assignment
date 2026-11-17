import numpy as np
import matplotlib.pyplot as plt


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    row_payoffs = row_matrix @ col_strategy
    col_payoffs = row_strategy @ col_matrix
    return np.array([
        row_payoffs.max() - row_strategy @ row_payoffs,
        col_payoffs.max() - col_payoffs @ col_strategy,
    ])


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    deltas = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.sum(deltas)


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    return compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy) / 2


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    p = np.full(row_matrix.shape[0], 1 / row_matrix.shape[0])
    q = np.full(col_matrix.shape[1], 1 / col_matrix.shape[1])
    last_row = np.zeros(row_matrix.shape[0])
    last_col = np.zeros(col_matrix.shape[1])
    strategies = []

    for t in range(num_iters):
        s = last_col if naive and t > 0 else q
        r = last_row if naive and t > 0 else p

        row_strategy = np.zeros(row_matrix.shape[0])
        col_strategy = np.zeros(col_matrix.shape[1])
        row_strategy[np.argmax(row_matrix @ s)] = 1.0
        col_strategy[np.argmax(r @ col_matrix)] = 1.0

        p += (row_strategy - p) * (1 / (t + 1))
        q += (col_strategy - q) * (1 / (t + 1))
        strategies.append((p.copy(), q.copy()))
        last_row = row_strategy
        last_col = col_strategy

    return strategies


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    values = [compute_exploitability(row_matrix, col_matrix, *strategy)
              for strategy in strategies]
    plt.plot(range(1, len(values) + 1), values, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Exploitability')
    return values


def main() -> None:
    row_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    col_matrix = -row_matrix
    strategies = fictitious_play(row_matrix, col_matrix, 1000, False)
    naive_strategies = fictitious_play(row_matrix, col_matrix, 1000, True)
    plot_exploitability(row_matrix, col_matrix, strategies, 'Fictitious Play')
    plot_exploitability(row_matrix, col_matrix, naive_strategies, 'Naive FP')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
