import numpy as np
import scipy
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

    plt.figure()
    for row in cmat:
        plt.plot(np.abs(row))
    plt.show()

def test_rfft2():
    img_in = scipy.misc.ascent().astype(np.float32)
    spc = mkl_dfti.rfft2(img_in)
    img_out = mkl_dfti.irfft2(spc)
    assert np.allclose(img_in, img_out, atol=1e-4)

    plt.figure()
    plt.imshow(img_out, cmap='gray')
    plt.show()

def test_rfft2_many():
    img_orig = scipy.misc.ascent().astype(np.float32)

    # demonstrate that the loopaxis can be any axis

    howmany = 5
    axes = list(range(3))
    loopaxis = axes.pop(1)

    # stack the 2d-image
    # then shift the loopaxis to some other position

    img_in = np.tile(img_orig, (howmany,1,1))
    img_in = np.rollaxis(img_in, 0, loopaxis+1)
    img_in = img_in.copy()

    # rfft2 use numpy to verify we are doing the same axes

    spc = np.fft.rfft2(img_in, axes=axes).astype(np.complex64)
    print(spc.shape)
    img_out = mkl_dfti.irfft2(spc, axes=axes)
    assert np.allclose(img_in, img_out, atol=1e-4)

    disp_img = img_out.take((0,), axis=loopaxis).squeeze()
    plt.figure()
    plt.imshow(disp_img, cmap='gray')
    plt.show()

def test_shapes():
    ttt = np.linspace(0, 1, 1000, endpoint=False)
    rmat = np.zeros((10,ttt.size), dtype=np.float32)

    for idx, row in enumerate(rmat):
        freq = (idx+1)*10
        row[:ttt.size] = np.sin(2*np.pi*freq*ttt)

    cmat = mkl_dfti.fft(rmat, 1500)

    plt.figure()
    for row in cmat:
        plt.plot(np.abs(row))
    plt.show()

if __name__=='__main__':
    test_shapes()
    test_r2c_inplace()
    test_rfft2()
    test_rfft2_many()

