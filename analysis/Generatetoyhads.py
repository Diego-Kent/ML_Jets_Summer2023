import numpy as np
import random
import matplotlib.pyplot as plt
import os
from silx.io.dictdump import dicttoh5
output_dir = './Output'
filename = 'Thadronsamples.h5'

# Define Hadronization function
im_dim = 16
num_samples = 10000
output_dir = 'Output'
def choose_z():
  z =[]
  N = random.randint(2,6)
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
#Define two different looking starting jets
Point = np.zeros((im_dim, im_dim))
Point[6,6] = 1
plt.imshow(Point, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'P.pdf'))
plt.clf()

#Define two different looking starting jets
Square = np.zeros((im_dim, im_dim), dtype=int)
index =[i+int(im_dim/4) for i in range(int(im_dim/2)) if i%2 ==0 ]
index_2 = [i+int(im_dim/2) for i in range(int(im_dim/2)) if i%2 ==0 ]
for i in range(len(index)):
  Square[index[i],int(im_dim/4)] = 1
  Square[int(im_dim/4),index[i]] = 1
  Square[index[i],int(im_dim*3/4)] = 1
  Square[int(im_dim*3/4),index[i]] = 1
plt.imshow(Square, cmap='gray')  # 'gray' colormap will display the image in black and white
plt.axis('off')  # Turn off the axis ticks and labels
plt.savefig(os.path.join(output_dir, 'S.pdf'))
plt.clf()

#Create data set 
results = {}
P_list = [[hadronize(Point),Point] for i in range(num_samples)]
S_list = [[hadronize(Square),Square] for i in range(num_samples)]
Sample_list = P_list + S_list
Shufflehad_list = []
Conditions_list = []
for i in range(num_samples):
  x = random.randint(0,len(Sample_list)-1)
  Shufflehad_list.append(Sample_list[x][0])
  Conditions_list.append(Sample_list[x][1])
Hadronsarray = np.stack(Shufflehad_list)
Consarray = np.stack(Conditions_list)
print(Hadronsarray.shape,Consarray.shape)
results['Had'] = Hadronsarray 
results['Cond'] = Consarray
#results['constant'] = Conditionsarray
print(f'Writing results to {output_dir}/{filename}...')
dicttoh5(results, os.path.join(output_dir, filename), overwrite_data=True)
print('All done.')
#Plot some pairs to see everithing ok
for k in range(5):
  plt.imshow(Hadronsarray[k], cmap='gray')  # 'gray' colormap will display the image in black and white
  plt.axis('off')  # Turn off the axis ticks and labels 
  plt.savefig(os.path.join(output_dir, f"Had{k}.pdf"))
  plt.clf()
  plt.imshow(Consarray[k], cmap='gray')  # 'gray' colormap will display the image in black and white
  plt.axis('off')  # Turn off the axis ticks and labels 
  plt.savefig(os.path.join(output_dir, f"Cond{k}.pdf"))
  plt.clf()
  