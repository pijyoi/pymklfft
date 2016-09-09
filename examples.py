import numpy as np
import matplotlib.pyplot as plt

import mkl_dfti

def test_r2c_inplace():
    ttt = np.linspace(0, 1, 1000, endpoint=False)
    mat = np.zeros((10,ttt.size+2), dtype=np.float32)
    rmat = mat[:,:ttt.size]
    cmat = mat.view(np.complex64)

    for idx, row in enumerate(rmat):
        freq = (idx+1)*10
        row[:ttt.size] = np.sin(2*np.pi*freq*ttt)

    desc = mkl_dfti.builder(rmat, cmat, axes=(-1,))
    desc.computeForward(rmat, cmat)

    for row in cmat:
        plt.plot(np.abs(row))
    plt.show()

if __name__=='__main__':
    test_r2c_inplace()
