import numpy
for d in [5,10,15,20,25,30,35,40,45,50,75,100,125,150,175,200]:
  try:
    s=numpy.load('scores_%d_dims.npy'%d)
    print(d,numpy.mean(s))
  except:
    print('no data for',d)
