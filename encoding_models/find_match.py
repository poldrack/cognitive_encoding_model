import numpy

def find_match(i,p,data):
    """
    find the real image that most closely matches the index image
    ala Kay et al., 2008
    """
    corr_all=numpy.zeros(p.shape[0])
    for j in range(p.shape[0]):
        corr_all[j]=numpy.corrcoef(p[i,:],data[j,:])[0,1]
    corr_true=corr_all[i]
    corr_max=numpy.nanmax(corr_all)
    if corr_true==corr_max:
        print('success!')
        success=1
    else:
        success=0
    corr_rank=numpy.nanmean(corr_true>corr_all)
    return ([corr_true,corr_max,corr_rank,success])
