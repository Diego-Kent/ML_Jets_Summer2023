import numpy as np
import random
import matplotlib.pyplot as plt
import os

im_dim = 28
output_dir = 'Output'
def choose_z():
  z =[]
  N = random.randint(2,8)
  for k in range(N):
    if k == 0:
      z_0 = random.uniform(0, 0.9)
      z.append(z_0)
    elif k == N-1:
      z.append(1-sum(z))
    else:
      z.append(random.uniform(0, 1-sum(z)))
  return z
def hadronize(P):
  L =[]
  while len(L)<2:
    for i in range(im_dim):
      for j in range(im_dim):
        if P[i,j]>0:
          M = np.zeros((im_dim, im_dim))
          z = choose_z()
          for l in range(len(z)):
            eta = np.round(np.random.normal(i, 1, 1)[0]).astype(np.int32)  
            phi = np.round(np.random.normal(j, 1, 1)[0]).astype(np.int32)  
            if (eta in range(im_dim) and phi in range(im_dim)):
              M[eta,phi] = z[l]
          L.append(M) 
  return sum(L)

Point = np.zeros((im_dim, im_dim))
Point[14,14] = 1

plt.imshow(Point, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'P.pdf'))
plt.clf()
P = hadronize(Point)
plt.imshow(P, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'Phad.pdf'))
plt.clf()
Square = np.zeros((28, 28), dtype=int)
index =[i+7 for i in range(14) if i%2 ==0 ]
index_2 = [i+14 for i in range(14) if i%2 ==0 ]
for i in range(len(index)):
  Square[index[i],7] = 1
  Square[7,index[i]] = 1
  Square[index[i],21] = 1
  Square[21,index[i]] = 1
plt.imshow(Square, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'S.pdf'))
plt.clf()
h= hadronize(Square)
plt.imshow(h, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'Shad.pdf'))
plt.clf()