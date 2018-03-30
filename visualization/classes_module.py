import numpy
from visualization import vis_utils

# -------------------------
# Feed-forward network
# -------------------------
class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, Z):
        for l in self.layers:
            Z = l.forward(Z)
        return Z

    def gradprop(self, DZ):
        for l in self.layers[::-1]:
            DZ = l.gradprop(DZ)
        return DZ

    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R



# -------------------------
# ReLU activation layer
# -------------------------
class ReLU:

    def forward(self, X):
        self.Z = X > 0
        return X*self.Z

    def gradprop(self, DY):
        return DY*self.Z

    def relprop(self, R):
        return R

# -------------------------
# Fully-connected layer
# -------------------------
class Linear:

    def __init__(self, W, b):
        self.W = W
        self.B = b

    def forward(self, X):
        self.X = X
        return numpy.dot(self.X, self.W)+self.B

    def gradprop(self, DY):
        self.DY = DY
        return numpy.dot(self.DY, self.W.T)

    def relprop(self, R):
        V = numpy.maximum(0, self.W)
        Z = numpy.dot(self.X, V) + 1e-9
        S = R / Z
        C = numpy.dot(S, V.T)
        R = self.X * C
        return R


class FirstLinear(Linear):

    def __init__(self, W, b):
        self.W = W
        self.B = b

    def relprop(self, R):
        W, V, U = self.W, numpy.maximum(0, self.W), numpy.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + vis_utils.lowest, self.X * 0 + vis_utils.highest

        Z = numpy.dot(X, W) - numpy.dot(L, V) - numpy.dot(H, U) + 1e-9;
        S = R / Z
        R = X * numpy.dot(S, W.T) - L * numpy.dot(S, V.T) - H * numpy.dot(S, U.T)
        return R


# # -------------------------
# # Sum-pooling layer
# # -------------------------
# class Pooling:
#
#     def forward(self, X):
#         self.X = X
#         self.Y = 0.5*(X[:, ::2, ::2, :]+X[:, ::2, 1::2, :]+X[:, 1::2, ::2, :]+X[:, 1::2, 1::2, :])
#         return self.Y
#
#     def gradprop(self, DY):
#         self.DY = DY
#         DX = self.X*0
#         for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
#             DX[:, i::2, j::2, :] += DY*0.5
#         return DX
#
# # -------------------------
# # Convolution layer
# # -------------------------
# class Convolution:
#
#     def __init__(self,name):
#         wshape = map(int,list(name.split("-")[-1].split("x")))
#         self.W = numpy.loadtxt(name+'-W.txt').reshape(wshape)
#         self.B = numpy.loadtxt(name+'-B.txt')
#
#     def forward(self,X):
#
#         self.X = X
#         mb,wx,hx,nx = X.shape
#         ww,hw,nx,ny = self.W.shape
#         wy,hy       = wx-ww+1,hx-hw+1
#
#         Y = numpy.zeros([mb,wy,hy,ny],dtype='float32')
#
#         for i in range(ww):
#             for j in range(hw):
#                 Y += numpy.dot(X[:,i:i+wy,j:j+hy,:],self.W[i,j,:,:])
#
#         return Y+self.B
#
#     def gradprop(self,DY):
#
#         self.DY = DY
#         mb,wy,hy,ny = DY.shape
#         ww,hw,nx,ny = self.W.shape
#
#         DX = self.X*0
#
#         for i in range(ww):
#             for j in range(hw):
#                 DX[:,i:i+wy,j:j+hy,:] += numpy.dot(DY,self.W[i,j,:,:].T)
#
#         return DX
