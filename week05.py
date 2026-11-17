import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    strategy = np.maximum(regrets, 0)
    if np.sum(strategy) == 0:
        strategy = np.ones(len(regrets))
    return strategy / np.sum(strategy)


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_iters <= 0:
        return []

    r_regrets = np.zeros(row_matrix.shape[0])
    s_regrets = np.zeros(col_matrix.shape[1])
    p = regret_matching(r_regrets)
    q = regret_matching(s_regrets)
    r_avg = p.copy()
    s_avg = q.copy()
    strategies = [(p.copy(), q.copy())]

    for t in range(1, num_iters):
        row_payoffs = row_matrix @ q
        col_payoffs = p @ col_matrix
        r_regrets += row_payoffs - p @ row_payoffs
        s_regrets += col_payoffs - col_payoffs @ q

        p = regret_matching(r_regrets)
        q = regret_matching(s_regrets)
        r_avg += (p - r_avg) / (t + 1)
        s_avg += (q - s_avg) / (t + 1)
        strategies.append((r_avg.copy(), s_avg.copy()))

    return strategies


def main() -> None:
    from week03 import compute_exploitability, fictitious_play, plot_exploitability
    import matplotlib.pyplot as plt

    row_matrix = np.array([[1, -1], [-1, 1]])
    col_matrix = -row_matrix
    strategies = regret_minimization(row_matrix, col_matrix, 1000)
    fp_strategies = fictitious_play(row_matrix, col_matrix, 1000, False)
    rm_value = compute_exploitability(row_matrix, col_matrix, *strategies[-1])
    fp_value = compute_exploitability(row_matrix, col_matrix, *fp_strategies[-1])
    print('Regret matching:', rm_value)
    print('Fictitious play:', fp_value)
    plot_exploitability(row_matrix, col_matrix, strategies, 'Regret Matching')
    plot_exploitability(row_matrix, col_matrix, fp_strategies, 'Fictitious Play')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
