import numpy
import matplotlib.pyplot as plt

scores={}
for d in [5,10,15,20,25,30,35,40,45,50,75,100,125,150,175,200]:
  try:
    s=numpy.load('cv_scores/scores_%d_dims.npy'%d)
    scores[d]=numpy.mean(s)
    print(d,scores[d])
  except:
    print('no data for',d)

dvals=list(scores.keys())
dvals.sort()

plt.plot(dvals,[scores[x] for x in dvals])
plt.show()
