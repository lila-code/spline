import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def sort_knots(knots):
    return knots[np.argsort(knots[:, 0])]


def h(xcoord):
    return [(x1 - x0) for x0, x1 in list(zip(xcoord, xcoord[1:]))]


def sigma(sigmas):
    """[Fonction qui génère la matrice sigma]

    Args:
        sigmas (tableau de réels): [contient les écarts_type]

    Returns:
        [type]: [description]
    """
    # S = np.zeros((n, n))
    # for i in range(n):
    #     S[i][i] = sigmas[i]
    # return S
    return np.diag(sigmas)


def Q(n, g):
    matq = np.zeros((n + 1, n - 1))
    g0 = 1 / g[0]
    matq[0][0] = g0
    matq[n][n - 2] = 1 / g[-1]
    matq[1][0] = -g0 - (1 / g[1])
    for i in range(1, n - 1):
        gi = 1 / g[i]
        matq[i][i] = gi
        matq[i + 1][i - 1] = matq[i][i]
        matq[i + 1][i] = -gi - (1 / g[i + 1])
    return matq


def D(c, h):
    return [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(0, len(c) - 1)]


def B(a, c, d, h):
    return [((a[i + 1] - a[i]) / h[i] - c[i] * h[i] - d[i] * h[i] * h[i]) for i in range(0, len(a))]


def A(y, sigma, q, c, p):
    """
    """
    return y - 1 / p * sigma ** 2 @ q @ y


def T(n, h):
    """[Fonction qui génère la matrice T]

    Args:
        n (entier): [ correspond à la taille de h ]
        h (vecteur de réels): [h est défini tel que h_i = x_i+1 - x_i pour 0 <= i <= n-1]

    Returns:
        [matrice] : [[n - 1][n - 1] de réels] 
    """
    # constuire la matrice T[n-1][n-1]
    # initialiser toutes les cases à 0
    T = np.zeros((n - 1, n - 1))
    # calculer les éléments de la diagonale et des deux bandes 
    for i in range(n - 2):
        T[i][i] = 2 * (h[i] + h[i + 1])
        T[i][i + 1] = h[i + 1]
        T[i + 1][i] = h[i + 1]
    T[n - 2][n - 2] = 2 * (h[n - 2] + h[n - 1])
    return 1 / 3 * T


def cholesky_2(A):
    n = len(A)
    # initialisation de la matrice L
    # L = [[0.0] * n for i in range(n)]
    L = np.zeros((n, n))

    # calcul de la decompostion de cholesky
    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            L[i][k] = np.sqrt(A[i][i] - tmp_sum) if i == k else (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L


def cholesky(n, A):
    """[Fonction qui calcule la factorisation de Cholsky d'une matrice, renvoie L et L.T]

    Args:
        n (entier): [dimesion de la matrice]
        A (matrice[n][n] de réels): [matrice à décomposer]

    Returns:
        [type]: [description]
    """
    n = len(A)
    for j in range(n):
        if A[j][j] <= 0:
            return False
        else:
            beta = np.sqrt(A[j][j])
            A[j][j] = beta
            for k in range(j + 1, n):
                A[k][j] = A[k][j] / beta
            for l in range(j + 1, n):
                for k in range(l, n):
                    A[k][l] = A[k][l] - A[k][j] * A[l][j]
    # print(A)
    # print(A.T)
    return A, A.T


def resolution_systeme_cholesky(A, b):
    """[Fonction qui résout Ax=b en sachant que A est symetrique definie positive]

    Args:
        n (entier): [dimension de la matrice]
        A (matrice[n][n] de réels): [matrice]
        b (vecteur de réels[n]): [membre droit]

    Returns:
        [type]: [description]
    """
    # méthodologie = https://i.imgur.com/ij5ZXKL.png
    matrix_l = cholesky_2(A)
    # Lnp = np.linalg.cholesky(A)
    # Lsp = sp_cholesky(A)
    # L, L_T = cholesky(n, A)
    # print(f'L2: {L2}')
    # print(f'L: {L}')
    # print(f'Lnp: {Lnp}')
    # print(f'Lsp: {Lsp}')
    return la.solve(matrix_l.T, la.solve(matrix_l, b))


def c(Q, sigma, T, p, y):
    """[Fonction qui résout (Q.T Σ^2 Q + pT )c = (p Q.T y)]

    Returns:
        [c]: [vecteur de réels[n]]
    """
    # print(f"Q {np.shape(Q)}")
    # print(f"sigma {np.shape(sigma)}")
    # print(f"T {np.shape(T)}")
    # print(f"Q.T {np.shape(Q.T)}")
    # test = ((Q.T) @ (sigma**2)) @ Q + p * T
    # print(f"Q.T @ sigma**2 @ Q {test}")
    A = Q.T @ sigma ** 2 @ Q + p * T
    b = p * (Q.T @ y)
    x1 = resolution_systeme_cholesky(A, b)
    # x2 = np.linalg.solve(A, b)
    return x1


def fct(x):
    """ test """
    return np.sin(x)


def d(c, h):
    return [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(0, len(h))]


def b(a, c, d, h):
    return [((a[i + 1] - a[i]) / h[i] - c[i] * h[i] - d[i] * h[i] * h[i]) for i in range(len(h))]


def a(y, sigma, q, c, p):
    # print(f"q {np.shape(q)}")
    # print(f"sigma**2 @ q {np.shape(sigma**2 @ q)}")
    # print(f"c {np.shape(c)}")
    # print(f"y {np.shape(y)}")
    return y - ((1 / p) * (sigma ** 2 @ q @ c))


def read_coordinate(filename):
    """
    pas de gestion d'erreur pour l'instant
    :param filename:
    :return:
    """
    current_directory = os.path.dirname(os.path.realpath(__file__))
    coordinates = os.path.join(current_directory, filename)
    points = []
    with open(coordinates, "r") as f:
        for line in f:
            row = line.split()
            points.append((float(row[0]), float(row[1])))
    return np.array(points)


if __name__ == "__main__":

    # noeuds = np.array([(3, 4), (1, 2), (0, 0), (6, 2)])
    # sigmas = [1, 1, 1, 1]
    # n = len(noeuds)
    # """[test de la fonction qui génère les écarts h]
    # """
    # noeuds = sort_knots(noeuds)
    # #print(noeuds)
    # h = h(noeuds)
    # #print(h)
    # """[test du code de la fonction qui génère la matrice des sigmas]
    # """
    # s = sigma(n, sigmas)
    # #print(s)

    # """[test de la fonction qui fait la décomposition de Cholesky]
    # """
    # L , LT = cholesky(n, s)
    # #print(L*LT)

    # """[test de la fonction qui résout le système linéaire avec décompostion Cholesky]
    # """
    # b = [1]*n
    # #print(b)
    # x = resolution_systeme_cholesky(n, s, b)
    # print(x)

    # noeuds = np.array([(x * np.pi / 4, np.cos(x * np.pi / 4)) for x in range(10)])
    # XXX: fichiers:
    # coordinate_{1..5}.txt -> sin, sin, cos, exp, cos
    noeuds = read_coordinate('coordinate_1.txt')
    nbnoeuds = len(noeuds)
    vdefault = 1
    n = nbnoeuds - 1
    # tri des noeuds selon leur coordonnées x
    noeuds = sort_knots(noeuds)
    xcoord = noeuds[:, 0]
    ycoord = noeuds[:, 1]
    # calcul des h
    h = h(xcoord)
    # génération de la matrice sigma
    sigma = np.diag([vdefault] * nbnoeuds)
    # génération de la matrice T
    t = T(n, h)
    # génération de la matrice Q 
    q = Q(n, h)
    # trouver le vecteur c

    lesp = [1e-5, 0.5, 1e5]  # TODO: trouver un p optimal par un algo
    # affichage de la spline résultat
    spline = lambda x, i: vecD[i] * (x - noeuds[i, 0]) ** 3 + vecC[i] * (x - noeuds[i, 0]) ** 2 + vecB[i] * (
            x - noeuds[i, 0]) + vecA[i]
    plt.scatter(xcoord, ycoord)
    for p in lesp:
        vecC = c(q, sigma, t, p, ycoord)
        vecA = a(ycoord, sigma, q, vecC, p)
        vecC = [0, *vecC, 0]
        vecD = d(vecC, h)
        vecB = b(vecA, vecC, vecD, h)

        # print(f"T = {3 * t}")
        # print(f"vecteur C = {vecC}")
        # print(f"vecteur A = {vecA}")
        # print(f"vecteur D = {vecD}")
        # print(f"vecteur B = {vecB}")

        for j in range(n):
            xval = xcoord[j]
            xarray = np.linspace(xval, xcoord[j + 1], n)
            yarray = spline(xarray, j)
            yval = spline(xval, j)
            # print(f'x, y = {xval:+.10f}, {yval:+.10f}')
            plt.plot(xarray, yarray)
    plt.show()
