import numpy as np
import math


def ex1b():
    A = np.array([[1, 2, 3, 4],
                  [2, 4, -4, 8],
                  [-5, 4, 1, 5],
                  [5, 0, -3, -7]])
    AT = A.transpose()
    ATA = AT @ A
    eigen_vals, w = np.linalg.eig(ATA)

    print("eigen_vals : \n%s " % str(eigen_vals))
    print("ATA : \n %s" % str(ATA))
    max_sigma = math.sqrt(max(eigen_vals))
    print("max sigma : %s" % max_sigma)
    x = w @ np.array([[1], [0], [0], [0]])
    print("x : %s" % str(x))


def gram_schmidt(A, make_ortogonal):
    a_cols = A.transpose()
    n=len(A[0])
    new_cols = np.array([a_cols[0] / np.linalg.norm(a_cols[0], ord=2)])
    rs = [np.zeros(n)]
    rs[0][0] =  np.linalg.norm(a_cols[0], ord=2)
    a_cols = np.delete(a_cols, 0, 0)

    for a in a_cols:
        q, r = make_ortogonal(a, new_cols,n)
        new_cols = np.vstack([new_cols, q])
        rs.append(r)
    finale_R = np.stack(rs,axis=0)
    return new_cols.transpose() , finale_R.transpose()


# def gram_schmidt2(A):
#     n=len(A)
#     R = np.zeros((n,n))
#     a_cols = A.transpose()
#     r[0][0] = np.linalg.norm(a_cols[0],ord=2)
#     final_q = [a_cols[0]/r[0][0]]
#     for i in range(1,len(a_cols)):
#         ai = a_cols[i]
#         qi=ai
#         for j in range(i-1):
#             qj = final_q[j]
#             R[j][i] = qj.transpose()@ai
#             qi=qi-R[j][i]*qj
#         R[i][i] = np.linalg.norm(qi,ord=2)
#         qi=qi/R[i][i]
#         final_q = np.vstack([final_q,qi])
#     return final_q.transpose(),R


def get_ortogonal_vector(vec, mat,rlen):
    '''
    this function caculate the ortogonal vector acording to the *original*
    gram_schmidt algoritem
    :param vec: the vector we want to make ortogonal to the other vectors
    :param mat: all past vectros
    :return:v2: the ortogonal vector
    '''
    v2 = vec
    r = np.zeros(rlen)
    j = 0
    for v1 in mat:
        r[j] = v1.transpose() @ vec
        v2 = v2 - r[j] * v1
        j += 1
    r[j] = np.linalg.norm(v2, ord=2)
    q = v2 / r[j]
    return q, r


def get_ortogonal_vector_modified(vec, mat,rlen):
    '''
    this function caculate the ortogonal vector acording to the *modified*
    gram_schmidt algoritem
    :param vec: the vector we want to make ortogonal to the other vectors
    :param mat: all past vectros
    :return:v2: the ortogonal vector
    '''
    v2 = vec
    r = np.zeros(rlen)
    j = 0
    for v1 in mat:
        r[j] = v1.transpose() @ v2
        v2 = v2 - r[j] * v1
        j += 1
    r[j] = np.linalg.norm(v2, ord=2)
    q = v2 /r[j]
    return q, r


def gram_schmidt_original(A):
    return gram_schmidt(A, get_ortogonal_vector)


def gram_schmidt_modified(A):
    return gram_schmidt(A, get_ortogonal_vector_modified)


def ex5(eps = 1):
    A = np.array([[1, 1, 1], [eps, 0, 0], [0, eps, 0], [0, 0, eps]])
    q1,r1 = gram_schmidt_original(A)
    print("\n==============q================= - gram_schmidt_original \n")
    print (q1)
    print("\n==============r================= - gram_schmidt_original\n")
    print(r1)

    print ("\n===========Q@R============== gram_schmidt_original\n")
    print (q1@r1)
    qtq = q1.transpose() @ q1
    print ("\n||QTQ-I|| - gram_schmidt_original: %s"%str(np.linalg.norm(qtq-np.identity(len(qtq[0])),ord='fro')))

    q2,r2 = gram_schmidt_modified(A)
    print("\n==============q================= - gram_schmidt_modified\n")
    print (q2)
    print("\n==============r================= - gram_schmidt_modified\n")
    print(r2)

    print ("\n===========Q@R============== gram_schmidt_modified\n")
    print (q2@r2)

    qtq = q2.transpose() @ q2
    print ("\n||QTQ-I|| - gram_schmidt_modified : %s"%str(np.linalg.norm(qtq-np.identity(len(qtq[0])),ord='fro')))


if __name__ == '__main__':
    ex5()
    ex5(float('1e-10'))



