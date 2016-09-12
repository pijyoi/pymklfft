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
        return False

    return True

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

def canonical_axes(ndim, axes):
    if axes is None:
        return list(range(ndim))
    axes = [(x+ndim)%ndim for x in axes]  # fix negative axes
    axes = list(set(axes))          # remove duplicates
    axes.sort()
    return axes

def canonical_axes_fftlens(ary, axes, fftlens):
    ndim = ary.ndim
    if axes is None:
        axes = list(range(ndim))
    axes = [(x+ndim)%ndim for x in axes]  # fix negative axes
    if len(set(axes)) != len(axes):
        raise ValueError("duplicate axes")

    if fftlens is None:
        fftlens = [ary.shape[ax] for ax in axes]
    if len(axes) != len(fftlens):
        raise ValueError("axes and fftlens have different lengths")

    # sort axes and make shapes follow the new sorted order
    axes_fftlens = zip(axes, fftlens)
    axes, fftlens = list(zip(*sorted(axes_fftlens)))

    # construct full shape tuple
    newshape = list(ary.shape)
    for ax,sh in zip(axes, fftlens):
        newshape[ax] = sh

    return axes, newshape

def builder(iarray, oarray, axes):
    if np.iscomplexobj(iarray) and np.iscomplexobj(oarray):
        domain = lib.DFTI_COMPLEX
        fftshape = iarray.shape
    elif np.isrealobj(iarray) and np.iscomplexobj(oarray):
        domain = lib.DFTI_REAL
        fftshape = iarray.shape
    elif np.iscomplexobj(iarray) and np.isrealobj(oarray):
        domain = lib.DFTI_REAL
        fftshape = oarray.shape
    else:
        raise ValueError("unsupported combination of array types")

    axes = canonical_axes(iarray.ndim, axes)
    loopaxes = [x for x in range(iarray.ndim) if x not in axes]
    if len(loopaxes) > 1:
        raise ValueError("unsupported axes")
    in_place = iarray.ctypes.data==oarray.ctypes.data

    lengths = list(fftshape)
    istrides = [x//iarray.itemsize for x in iarray.strides]
    ostrides = [x//oarray.itemsize for x in oarray.strides]

    if not loopaxes:
        howmany = 1
    else:
        loopaxis = loopaxes[0]
        howmany = lengths.pop(loopaxis)
        idist = istrides.pop(loopaxis)
        odist = ostrides.pop(loopaxis)

    precision = ndarray_precision(iarray)
    desc = DftiDescriptor(precision, domain)
    desc.create(lengths)

    desc.setInputStrides(istrides)
    desc.setOutputStrides(ostrides)

    if howmany > 1:
        desc.setValueInt(lib.DFTI_NUMBER_OF_TRANSFORMS, howmany)
        desc.setInputDistance(idist)
        desc.setOutputDistance(odist)

    desc.setValueFloat(lib.DFTI_BACKWARD_SCALE, 1.0/np.product(lengths))

    desc.setInPlace(in_place)

    if domain==lib.DFTI_REAL:
        # use CCE storage format
        desc.setValueInt(lib.DFTI_CONJUGATE_EVEN_STORAGE, lib.DFTI_COMPLEX_COMPLEX)

    desc.commit()
    return desc

def rfftnd_helper(array_in, dirn, axes=None, fftlens=None):
    axes, fftshape = canonical_axes_fftlens(array_in, axes, fftlens)
    realaxis = axes[-1]

    iarray = array_in
    rshape = list(fftshape)
    cshape = list(fftshape)

    if np.isrealobj(iarray):
        assert dirn==FORWARD
        cshape[realaxis] = rshape[realaxis]//2 + 1

        ishape, oshape, odtype = rshape, cshape, r2c_dst_dtype(iarray.dtype)

    else:
        assert dirn==BACKWARD

        if fftlens is None:
            # user didn't specify fftlens
            # therefore fftshape is the shape of the complex input array
            rshape[realaxis] = 2*(cshape[realaxis]-1)
        else:
            # user specified fftlens
            # for irfft, that specifies the realfftlen
            cshape[realaxis] = rshape[realaxis]//2 + 1

        ishape, oshape, odtype = cshape, rshape, c2r_dst_dtype(iarray.dtype)

    slices = [slice(min(ishape[ax], iarray.shape[ax])) for ax in range(iarray.ndim)]
    if all([ishape[ax] <= iarray.shape[ax] for ax in axes]):
        iarray = iarray[slices]
    else:
        array_tmp = np.zeros(ishape, dtype=iarray.dtype)
        array_tmp[slices] = iarray[slices]
        iarray = array_tmp

    oarray = np.empty(oshape, odtype)
    compute_args = (dirn, iarray, oarray)

    desc = builder(iarray, oarray, axes)
    desc.compute(*compute_args)

    return oarray

def fftnd_helper(array_in, dirn, axes=None, fftlens=None):
    axes, fftshape = canonical_axes_fftlens(array_in, axes, fftlens)

    slices = [slice(min(fftshape[ax], array_in.shape[ax])) for ax in range(array_in.ndim)]
    if all([fftshape[ax] <= array_in.shape[ax] for ax in axes]):
        array_in = array_in[slices]
    else:
        array_tmp = np.zeros(fftshape, dtype=array_in.dtype)
        array_tmp[slices] = array_in[slices]
        array_in = array_tmp

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

    desc = builder(iarray, oarray, axes)
    desc.compute(*compute_args)

    return oarray

def rfft(a, n=None, axis=-1):
    s = None if n is None else (n,)
    return rfftnd_helper(a, FORWARD, (axis,), s)

def irfft(a, n=None, axis=-1):
    s = None if n is None else (n,)
    return rfftnd_helper(a, BACKWARD, (axis,), s)

def rfft2(a, s=None, axes=(-2,-1)):
    return rfftnd_helper(a, FORWARD, axes, s)

def irfft2(a, s=None, axes=(-2,-1)):
    return rfftnd_helper(a, BACKWARD, axes, s)

def rfftn(a, s=None, axes=None):
    return rfftnd_helper(a, FORWARD, axes, s)

def irfftn(a, s=None, axes=None):
    return rfftnd_helper(a, BACKWARD, axes, s)

def fft(a, n=None, axis=-1):
    s = None if n is None else (n,)
    return fftnd_helper(a, FORWARD, (axis,), s)

def ifft(a, n=None, axis=-1):
    s = None if n is None else (n,)
    return fftnd_helper(a, BACKWARD, (axis,), s)

def fft2(a, s=None, axes=(-2,-1)):
    return fftnd_helper(a, FORWARD, axes, s)

def ifft2(a, s=None, axes=(-2,-1)):
    return fftnd_helper(a, BACKWARD, axes, s)

def fftn(a, s=None, axes=None):
    return fftnd_helper(a, FORWARD, axes, s)

def ifftn(a, s=None, axes=None):
    return fftnd_helper(a, BACKWARD, axes, s)

