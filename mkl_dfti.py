from __future__ import print_function, division, absolute_import

import os
import sys
import numpy as np
import cffi

ffi = cffi.FFI()
ffi.cdef("""
enum DFTI_CONFIG_PARAM
{
    DFTI_FORWARD_SCALE  = 4,
    DFTI_BACKWARD_SCALE = 5,
    DFTI_NUMBER_OF_TRANSFORMS = 7,
    DFTI_CONJUGATE_EVEN_STORAGE = 10,
    DFTI_PLACEMENT = 11,
    DFTI_INPUT_STRIDES = 12,
    DFTI_OUTPUT_STRIDES = 13,
    DFTI_INPUT_DISTANCE = 14,
    DFTI_OUTPUT_DISTANCE = 15,
};

enum DFTI_CONFIG_VALUE
{
    DFTI_COMPLEX = 32,
    DFTI_REAL = 33,
    DFTI_SINGLE = 35,
    DFTI_DOUBLE = 36,
    DFTI_COMPLEX_COMPLEX = 39,
    DFTI_INPLACE = 43,
    DFTI_NOT_INPLACE = 44,
};

typedef long MKL_LONG;
struct DFTI_DESCRIPTOR;
typedef struct DFTI_DESCRIPTOR *DFTI_DESCRIPTOR_HANDLE;

MKL_LONG DftiCreateDescriptor(DFTI_DESCRIPTOR_HANDLE*,
                              enum DFTI_CONFIG_VALUE, /* precision */
                              enum DFTI_CONFIG_VALUE, /* domain */
                              MKL_LONG, ...);
MKL_LONG DftiCopyDescriptor(DFTI_DESCRIPTOR_HANDLE, /* from descriptor */
                            DFTI_DESCRIPTOR_HANDLE*); /* to descriptor */
MKL_LONG DftiCommitDescriptor(DFTI_DESCRIPTOR_HANDLE);
MKL_LONG DftiComputeForward(DFTI_DESCRIPTOR_HANDLE, void*, ...);
MKL_LONG DftiComputeBackward(DFTI_DESCRIPTOR_HANDLE, void*, ...);
MKL_LONG DftiSetValue(DFTI_DESCRIPTOR_HANDLE, enum DFTI_CONFIG_PARAM, ...);
MKL_LONG DftiGetValue(DFTI_DESCRIPTOR_HANDLE, enum DFTI_CONFIG_PARAM, ...);
MKL_LONG DftiFreeDescriptor(DFTI_DESCRIPTOR_HANDLE*);
char* DftiErrorMessage(MKL_LONG);
MKL_LONG DftiErrorClass(MKL_LONG, MKL_LONG);
""")

lib = None
def install(dll_path=None):
    global lib

    dll_name = None
    if sys.platform.startswith("linux"):
        dll_name = "libmkl_rt.so"
    elif sys.platform=="win32":
        dll_name = "mkl_rt.dll"

    locations = []
    if dll_path is not None:
        locations.append(dll_path)
    else:
        # try some known locations
        # Anaconda Linux
        locations.append(os.path.join(sys.exec_prefix, "lib"))
        # WinPython
        locations.append(os.path.join(os.path.dirname(np.__file__), "core"))
        # MKL installed into our home directory
        locations.append(os.path.join(os.environ["HOME"], "intel/mkl/lib/intel64"))

    for dll_path in locations:
        dll_pathname = os.path.join(dll_path, dll_name)
        if os.path.exists(dll_pathname):
            try:
                lib = ffi.dlopen(dll_pathname)
                break
            except OSError as e:
                print(e)
    else:
        print("failed to load MKL libraries")

install()

class DftiError(Exception):
    pass

def error_message(rc):
    return ffi.string(lib.DftiErrorMessage(rc))

def raise_if_error(rc):
    if rc!=0:
        raise DftiError(error_message(rc))

def ndarray_precision(arr):
    char = arr.dtype.char
    if char in 'fF':
        return lib.DFTI_SINGLE
    elif char in 'dD':
        return lib.DFTI_DOUBLE
    else:
        raise ValueError("unsupported dtype")

class DftiDescriptor:
    def __init__(self, precision, domain):
        self.handle = ffi.new("DFTI_DESCRIPTOR_HANDLE*")

        self.precision = precision
        self.domain = domain

    def setValueInt(self, config, value):
        cvalue = ffi.cast("MKL_LONG", value)
        rc = lib.DftiSetValue(self.handle[0], config, cvalue)
        raise_if_error(rc)

    def setValueFloat(self, config, value):
        cvalue = ffi.cast("double", value)
        rc = lib.DftiSetValue(self.handle[0], config, cvalue)
        raise_if_error(rc)

    def setValueArray(self, config, values):
        cvalue = ffi.new("MKL_LONG[]", values)
        rc = lib.DftiSetValue(self.handle[0], config, cvalue)
        raise_if_error(rc)

    def setInputDistance(self, value):
        self.setValueInt(lib.DFTI_INPUT_DISTANCE, value)

    def setOutputDistance(self, value):
        self.setValueInt(lib.DFTI_OUTPUT_DISTANCE, value)

    def setInputStrides(self, value):
        # Note that MKL strides is of size rank+1
        # with the first element being an offset
        self.setValueArray(lib.DFTI_INPUT_STRIDES, (0,) + tuple(value))

    def setOutputStrides(self, value):
        self.setValueArray(lib.DFTI_OUTPUT_STRIDES, (0,) + tuple(value))

    def create(self, dims):
        if isinstance(dims, int):
            dims = (dims,)

        rank = len(dims)
        if rank==1:
            fftlen = ffi.cast("MKL_LONG", dims[0])
            rc = lib.DftiCreateDescriptor(self.handle, self.precision, self.domain, 1, fftlen)
        else:
            lengths = ffi.new("MKL_LONG[]", dims)
            rc = lib.DftiCreateDescriptor(self.handle, self.precision, self.domain, rank, lengths)
        raise_if_error(rc)

        if self.domain==lib.DFTI_REAL:
            # use CCE storage format
            self.setValueInt(lib.DFTI_CONJUGATE_EVEN_STORAGE, lib.DFTI_COMPLEX_COMPLEX)

    def commit(self):
        rc = lib.DftiCommitDescriptor(self.handle[0])
        raise_if_error(rc)

    def computeForward(self, src, dst=None):
        args = [ffi.cast("void*", src.ctypes.data)]
        if dst is not None:
            args.append(ffi.cast("void*", dst.ctypes.data))
        rc = lib.DftiComputeForward(self.handle[0], *args)
        raise_if_error(rc)

    def computeBackward(self, src, dst=None):
        args = [ffi.cast("void*", src.ctypes.data)]
        if dst is not None:
            args.append(ffi.cast("void*", dst.ctypes.data))
        rc = lib.DftiComputeBackward(self.handle[0], *args)
        raise_if_error(rc)

    def compute(self, dirn, src, dst=None):
        if dirn==FORWARD:
            self.computeForward(src, dst)
        else:
            self.computeBackward(src, dst)

    def setInPlace(self, value):
        # the library default is INPLACE
        arg = lib.DFTI_INPLACE if value else lib.DFTI_NOT_INPLACE
        self.setValueInt(lib.DFTI_PLACEMENT, arg)

    def __del__(self):
        rc = lib.DftiFreeDescriptor(self.handle)
        raise_if_error(rc)

FORWARD, BACKWARD = -1, +1

def r2c_dst_dtype(src_dtype):
    return np.dtype(src_dtype.char.upper())

def c2r_dst_dtype(src_dtype):
    return np.dtype(src_dtype.char.lower())

def get_realaxis(ndim, loopaxis):
    # real axis is the last axis after excluding the loopaxis.
    # NOTE: choice of last axis follows C row major convention
    allaxes = list(range(ndim))
    if loopaxis is not None:
        allaxes.pop(loopaxis)
    realaxis = allaxes[-1]
    return realaxis

def axes_to_loopaxis(ndim, axes):
    if axes is None:
        return None
    axes = [(x+ndim)%ndim for x in axes]  # fix negative axes
    work = set(range(ndim))
    work.difference_update(axes)
    if len(work)==0:
        return None
    elif len(work)==1:
        return work.pop()
    else:
        raise ValueError("unsupported axes")

def builder_real(iarray, oarray, axes):
    if np.isrealobj(iarray) and np.iscomplexobj(oarray):
        # r2c
        real_array = iarray
    elif np.isrealobj(oarray) and np.iscomplexobj(iarray):
        # c2r
        real_array = oarray
    else:
        raise ValueError("must have real and complex array")
    
    loopaxis = axes_to_loopaxis(iarray.ndim, axes)
    in_place = iarray.ctypes.data==oarray.ctypes.data
    
    lengths = list(real_array.shape)
    istrides = [x//iarray.itemsize for x in iarray.strides]
    ostrides = [x//oarray.itemsize for x in oarray.strides]

    if loopaxis is None:
        howmany = 1
    else:
        howmany = lengths.pop(loopaxis)
        idist = istrides.pop(loopaxis)
        odist = ostrides.pop(loopaxis)

    precision = ndarray_precision(iarray)
    desc = DftiDescriptor(precision, lib.DFTI_REAL)
    desc.create(lengths)

    desc.setInputStrides(istrides)
    desc.setOutputStrides(ostrides)

    if howmany > 1:
        desc.setValueInt(lib.DFTI_NUMBER_OF_TRANSFORMS, howmany)
        desc.setInputDistance(idist)
        desc.setOutputDistance(odist)

    desc.setValueFloat(lib.DFTI_BACKWARD_SCALE, 1.0/np.product(lengths))

    desc.setInPlace(in_place)
    
    desc.commit()
    return desc

def builder_complex(iarray, oarray, axes):
    loopaxis = axes_to_loopaxis(iarray.ndim, axes)
    in_place = iarray.ctypes.data==oarray.ctypes.data
    
    lengths = list(iarray.shape)
    istrides = [x//iarray.itemsize for x in iarray.strides]
    ostrides = [x//oarray.itemsize for x in oarray.strides]

    if loopaxis is None:
        howmany = 1
    else:
        howmany = lengths.pop(loopaxis)
        idist = istrides.pop(loopaxis)
        odist = ostrides.pop(loopaxis)

    precision = ndarray_precision(iarray)
    desc = DftiDescriptor(precision, lib.DFTI_COMPLEX)
    desc.create(lengths)

    desc.setInputStrides(istrides)
    desc.setOutputStrides(ostrides)

    if howmany > 1:
        desc.setValueInt(lib.DFTI_NUMBER_OF_TRANSFORMS, howmany)
        desc.setInputDistance(idist)
        desc.setOutputDistance(odist)

    desc.setValueFloat(lib.DFTI_BACKWARD_SCALE, 1.0/np.product(lengths))

    desc.setInPlace(in_place)

    desc.commit()
    return desc

def rfftnd_helper(array_in, dirn, axes=None):
    loopaxis = axes_to_loopaxis(array_in.ndim, axes)
    iarray = array_in
    ishape = iarray.shape
    realaxis = get_realaxis(iarray.ndim, loopaxis)

    if np.isrealobj(iarray):
        assert dirn==FORWARD
        rshape = list(ishape)
        cshape = list(ishape)
        cshape[realaxis] = cshape[realaxis]//2 + 1

        oarray = np.empty(cshape, r2c_dst_dtype(iarray.dtype))
        compute_args = (dirn, iarray, oarray)
    else:
        assert dirn==BACKWARD
        # FIXME: only works if original length was even
        rshape = list(ishape)
        rshape[realaxis] = 2*(rshape[realaxis]-1)
        cshape = list(ishape)

        oarray = np.empty(rshape, c2r_dst_dtype(iarray.dtype))
        compute_args = (dirn, iarray, oarray)

    desc = builder_real(iarray, oarray, axes)
    desc.compute(*compute_args)

    return oarray

def fftnd_helper(array_in, dirn, axes=None):
    # if array_in is real-valued, we copy the input to the complex output
    # then do an in-place fft on the complex output array
    if np.isrealobj(array_in):
        assert dirn==FORWARD
        iarray = array_in.astype(r2c_dst_dtype(array_in.dtype))
        oarray = iarray
        compute_args = (dirn, iarray)
    else:
        iarray = array_in
        oarray = np.empty_like(iarray)
        compute_args = (dirn, iarray, oarray)

    desc = builder_complex(iarray, oarray, axes)
    desc.compute(*compute_args)

    return oarray

def rfft(x, axis=-1):
    return rfftnd_helper(x, FORWARD, (axis,))

def irfft(x, axis=-1):
    return rfftnd_helper(x, BACKWARD, (axis,))

def rfft2(x, axes=(-2,-1)):
    return rfftnd_helper(x, FORWARD, axes)

def irfft2(x, axes=(-2,-1)):
    return rfftnd_helper(x, BACKWARD, axes)

def rfftn(x, axes=None):
    return rfftnd_helper(x, FORWARD, axes)

def irfftn(x, axes=None):
    return rfftnd_helper(x, BACKWARD, axes)

def fft(x, axis=-1):
    return fftnd_helper(x, FORWARD, (axis,))

def ifft(x, axis=-1):
    return fftnd_helper(x, BACKWARD, (axis,))

def fft2(x, axes=(-2,-1)):
    return fftnd_helper(x, FORWARD, axes)

def ifft2(x, axes=(-2,-1)):
    return fftnd_helper(x, BACKWARD, axes)

def fftn(x, axes=None):
    return fftnd_helper(x, FORWARD, axes)

def ifftn(x, axes=None):
    return fftnd_helper(x, BACKWARD, axes)

def test():
    import matplotlib.pyplot as plt

    ttt = np.linspace(0, 1, 1000, endpoint=False)
    mat = np.zeros((10,1000), dtype=np.float32)
    for row in range(mat.shape[0]):
        freq = (row+1)*10
        mat[row] = np.sin(2*np.pi*freq*ttt)

    # test "column" fft
    y = rfft(mat.T, axis=0)
    y = y.T

    for row in range(y.shape[0]):
        plt.plot(np.abs(y[row]))
    plt.show()

if __name__=='__main__':
    test()

