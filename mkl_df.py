import os
import sys
import numpy as np
import cffi

ffi = cffi.FFI()
ffi.cdef("""
typedef void* DFTaskPtr;

int dfdNewTask1D(DFTaskPtr *, int, const double[], int, int, const double[], int);
int dfdEditPPSpline1D(DFTaskPtr, int, int, int, const double[], int, const double[], const double[], int);
int dfdConstruct1D(DFTaskPtr, int, int);
int dfdInterpolate1D(DFTaskPtr, int, int, int, const double[], int, int, const int[], const double[], double[], int, int[]);
int dfDeleteTask(DFTaskPtr *);

#define DF_STATUS_OK 0

#define DF_NO_HINT 0
#define DF_NON_UNIFORM_PARTITION 1
#define DF_MATRIX_STORAGE_ROWS 16

#define DF_PP_CUBIC 4

#define DF_PP_NATURAL 2
#define DF_PP_BESSEL 4

#define DF_NO_BC 0
#define DF_BC_NOT_A_KNOT 1
#define DF_BC_FREE_END 2
#define DF_BC_PERIODIC 64

#define DF_NO_IC 0

#define DF_PP_SPLINE 0

#define DF_METHOD_STD 0
#define DF_METHOD_PP 1

#define DF_INTERP 1

#define DF_NO_APRIORI_INFO 0
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
        # Anaconda Windows
        locations.append(os.path.join(sys.exec_prefix, "Library/bin"))
        # WinPython
        locations.append(os.path.join(os.path.dirname(np.__file__), "core"))
        # MKL installed into our home directory
        if "HOME" in os.environ:
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

class DfError(Exception):
    pass

def error_message(rc):
    return "error code {}".format(rc)

def raise_if_error(rc):
    if rc!=lib.DF_STATUS_OK:
        raise DfError(error_message(rc))

class DfTask:
    def __init__(self, x, y, bc_type=lib.DF_BC_FREE_END):
        self.task = ffi.new("DFTaskPtr*")
        rc = lib.dfdNewTask1D(self.task, x.size, ffi.cast("double*", x.ctypes.data), lib.DF_NON_UNIFORM_PARTITION, 1, ffi.cast("double*", y.ctypes.data), lib.DF_NO_HINT)
        raise_if_error(rc)
        
        s_order = lib.DF_PP_CUBIC
        s_type = lib.DF_PP_NATURAL
        bc = ffi.NULL
        ic_type = lib.DF_NO_IC
        ic = ffi.NULL
        self.scoeff = np.empty((x.size-1)*s_order, dtype=x.dtype)
        scoeffhint = lib.DF_NO_HINT
        rc = lib.dfdEditPPSpline1D(self.task[0], s_order, s_type, bc_type, bc, ic_type, ic, ffi.cast("double*", self.scoeff.ctypes.data), scoeffhint)
        raise_if_error(rc)
        rc = lib.dfdConstruct1D(self.task[0], lib.DF_PP_SPLINE, lib.DF_METHOD_STD)
        raise_if_error(rc)

    def interpolate(self, site, ndorder=1):
        sitehint = lib.DF_NON_UNIFORM_PARTITION
        dorder = ffi.new("int[]", ndorder)
        dorder[ndorder-1] = 1
        # there is an error with Intel's example, datahint is a float array, just happens that DF_NO_APRIORI_INFO=0
        # datahint = lib.DF_NO_APRIORI_INFO
        datahint = ffi.NULL
        r = np.empty_like(site)
        rhint = lib.DF_MATRIX_STORAGE_ROWS
        cell = ffi.NULL
        rc = lib.dfdInterpolate1D(self.task[0], lib.DF_INTERP, lib.DF_METHOD_PP, site.size, ffi.cast("double*", site.ctypes.data), sitehint, ndorder, dorder, datahint, ffi.cast("double*", r.ctypes.data), rhint, cell)
        raise_if_error(rc)
        return r
    
    def __del__(self):
        rc = lib.dfDeleteTask(self.task)
        raise_if_error(rc)

class CubicSpline:
    def __init__(self, x, y, bc_type='not-a-knot'):
        if bc_type=='not-a-knot':
            bc_type=lib.DF_BC_NOT_A_KNOT
        elif bc_type=='periodic':
            bc_type=lib.DF_BC_PERIODIC
        elif bc_type=='clamped':
            pass
        elif bc_type=='natural':
            bc_type=lib.DF_BC_FREE_END
            
        self.dftask = DfTask(x, y, bc_type)
    
    def __call__(self, x, nu=0):
        return self.dftask.interpolate(x, ndorder=nu+1)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    # from scipy.interpolate import CubicSpline
    
    # 100 Hz sampling
    x0 = 2*np.pi*2*np.linspace(0, 1, 101)
    y0 = np.cos(x0)
    dy0 = -np.sin(x0)
    
    # 10 Hz sampling
    x1 = x0[::10].copy()
    y1 = y0[::10].copy()
    cs = CubicSpline(x1, y1, bc_type='natural')
    y = cs(x0)
    dy = cs(x0, 1)

    plt.plot(y0)
    plt.plot(dy0)
    plt.plot(y, '.')
    plt.plot(dy, '.')
    
    plt.show()
    
