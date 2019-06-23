from pprint import pprint

from numpy import *
from scipy.linalg import norm, inv


# task 1
# symmetric function if A = A.transpose

# short solution


def is_symmetric(M):
    if allclose(M, M.T):
        return 1
    if allclose(M.T, -M):
        return -1
    return 0


# classic solution

def is_symmetric2(M):
    for i in range(len(M)):
        for j in range(len(M[i])):

            if i != j:  # because when i = j it is always equal whether symmetric or skew symmetric
                if M[i][j] == M[j][i]:
                    return 1
                if M[i][j] == -M[j][i]:
                    return -1

    return 0


# X = array([[20, 120, 200], [120, 10, 150], [200, 150, 30]])  # symmetric
# Y = array([[0, 1, -3], [-1, 0, -2], [3, 2, 0]])  # skew-symmetric
# C = array([[1, 1, -1], [11, 2, 0], [4, 9, 5]])  # neither symmetric nor skew symmetric
#
# print(is_symmetric(X))
# print("classic solution", is_symmetric2(X))
#
# print(is_symmetric(Y))
# print("classic solution ", is_symmetric2(Y))
#
# print(is_symmetric(C))
# print("classic solution", is_symmetric2(C))


# task 2

def is_orthogonal(v1, v2):
    """
        function that checks if the given vectors are orthogonal.

        Args:
            ``v1``: first vector
            ``v2``: second vector

        Returns:
            ``boolean``: Whether given vectors are orthogonal or not.
    """

    if dot(v1, v2) == 0:
        return True
    else:
        return False


v1 = array([1, -2, 3])
v2 = array([5, 4, 1])
print(is_orthogonal(v1, v2))


# task 3

# matrix normalization using (2-norm)
def normalization(v):
    # calculate magnitude of the given vector
    v_magnitude = sqrt(sum(v ** 2))

    # generate an empty vector
    norm_vector = []

    for x in v:
        norm_vector.append(x / v_magnitude)

    return array(norm_vector)


v4 = array([1, 2, 3])
print(normalization(v4))


# matrix normalization using numpy.norm

def normalization_2(v):
    """
        function that normalizes each row of the matrix x to have unit length.

        Args:
         ``v``: A numpy matrix of shape (n, m)

        Returns:
         ``x``: A vector which is the normalized form of numpy matrix.
    """
    return v / norm(v)


print(normalization_2(v4))

# task 4
print("###########################################################")


def inverse_test(alpha):
    A = array([[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]])
    print('Matrix A is: ', A)
    B = inv(A)

    print('AB is = ', dot(A, B))
    print('BA is = ', dot(B, A))

    print('Transpose of A is ', A.T)


inverse_test(10)

# task 5
Z = zeros(20 * 20).reshape(20, 20)
i, j = indices(Z.shape)
Z[i == j] = 4
Z[i == j+1] = 1
Z[i == j -1] = 1
#EIG = sl.eig(Z)
