from math import * 

#################   Vectors

class vector(list):
  def __add__(self, other):
    return self.__class__(map(lambda x,y: x+y, self, other)) #self.__class__ is vector, unless i am a polynomial or something!

  def __neg__(self):
    return self.__class__(map(lambda x: -x, self))

  def __sub__(self, other):
    return self.__class__(map(lambda x,y: x-y, self, other))

  def __mul__(self, other): #mult by scalar
    return self.__class__(map(lambda x: x*other, self))

  def __rmul__(self, other):
    return (self*other)

  def norm_sq(self):
    return dot(self, self)

  def norm(self):
    return sqrt(self.norm_sq())
    
  def normalised(self):
    my_norm = self.norm()
    assert my_norm != 0, 'norm of vector is zero, cannot normalise'
    return (1.0/my_norm) * self

def cross(a, b):
  return vector([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def dot(a,b):
  return sum( [ a[i]*b[i] for i in range(len(a)) ] )

#################   Matrices

def zero_matrix(m,n):
  """create zero matrix"""
  new_matrix = [[0 for row_entry in range(n)] for row in range(m)]
  return new_matrix

def matrix_mult(matrix1,matrix2):
  """matrix multiplication"""
  m1, m2 = matrix1, matrix2
  assert len(m1[0]) == len(m2), 'Matrices must be m*n and n*p to multiply!'

  new_matrix = zero_matrix(len(m1),len(m2[0]))
  for i in range(len(m1)):
    for j in range(len(m2[0])):
      for k in range(len(m2)):
        new_matrix[i][j] += m1[i][k]*m2[k][j]
  return new_matrix 

def matrix_mult_vector(M,v): 
  """apply matrix M to vector v, treated as a vertical vector"""
  u = [[entry] for entry in v]
  out = matrix_mult(M, u)
  return [entry for [entry] in out]

def matrix2_det(M):
  """determinant of 2x2 matrix"""
  return M[0][0]*M[1][1] - M[0][1]*M[1][0]
  
def matrix2_inv(M):
  """inverse of 2x2 matrix"""
  inv_det = 1.0/matrix2_det(M)
  return [[inv_det*M[1][1],-inv_det*M[0][1]],[-inv_det*M[1][0],inv_det*M[0][0]]]
