import numpy as np
from scipy.optimize import linprog


def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = row_matrix.shape
    p_lp = linprog(
        np.append(np.zeros(m), -1.0),
        A_ub=np.column_stack((-row_matrix.T, np.ones(n))),
        b_ub=np.zeros(n),
        A_eq=np.append(np.ones(m), 0.0)[None, :],
        b_eq=[1],
        bounds=[(0, None)] * m + [(None, None)],
    )
    q_lp = linprog(
        np.append(np.zeros(n), 1.0),
        A_ub=np.column_stack((row_matrix, -np.ones(m))),
        b_ub=np.zeros(m),
        A_eq=np.append(np.ones(n), 0.0)[None, :],
        b_eq=[1],
        bounds=[(0, None)] * n + [(None, None)],
    )
    if not p_lp.success or not q_lp.success:
        raise RuntimeError('Could not solve zero-sum game')
    return p_lp.x[:m], q_lp.x[:n]


def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    m, n = row_matrix.shape
    a_ub = []
    for i in range(m):
        for j in range(m):
            constraint = np.zeros((m, n))
            constraint[i, :] = row_matrix[j] - row_matrix[i]
            a_ub.append(constraint.ravel())
    for i in range(n):
        for j in range(n):
            constraint = np.zeros((m, n))
            constraint[:, i] = col_matrix[:, j] - col_matrix[:, i]
            a_ub.append(constraint.ravel())

    result = linprog(
        np.zeros(m * n),
        A_ub=np.array(a_ub),
        b_ub=np.zeros(len(a_ub)),
        A_eq=np.ones((1, m * n)),
        b_eq=[1],
        bounds=[(0, None)] * (m * n),
    )
    if not result.success:
        raise RuntimeError('Could not find correlated equilibrium')
    return result.x.reshape(m, n)


def main() -> None:
    row_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    p, q = find_nash_equilibrium(row_matrix)
    print(p, q)
    print('Correlated equilibrium:')
    print(find_correlated_equilibrium(row_matrix, -row_matrix))


if __name__ == '__main__':
    main()
