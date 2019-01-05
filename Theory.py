import numpy
import theano.tensor as T
from theano import function
from theano import pp

print("Hello world")

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
print(pp(z))
# f = function([x, y], z)
numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)
