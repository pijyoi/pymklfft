import os
import sys
import numpy as np
import cffi

ffi = cffi.FFI()
ffi.cdef("""
typedef void* DFTaskPtr;

int dfdNewTask1D(DFTaskPtr *, int, const double[], int, int, const double[],
                 int);
int dfdEditPPSpline1D(DFTaskPtr, int, int, int, const double[], int,
                      const double[], const double[], int);
int dfdConstruct1D(DFTaskPtr, int, int);
int dfdInterpolate1D(DFTaskPtr, int, int, int, const double[], int, int,
                     const int[], const double[], double[], int, int[]);
int dfDeleteTask(DFTaskPtr *);
int dfdEditPtr(DFTaskPtr, int, const double[]);
int dfiEditVal(DFTaskPtr, int, int);

#define DF_STATUS_OK 0

#define DF_NO_HINT 0
#define DF_NON_UNIFORM_PARTITION 1
#define DF_MATRIX_STORAGE_ROWS 16

#define DF_IC 3
#define DF_BC 4
#define DF_IC_TYPE 20
#define DF_BC_TYPE 21

#define DF_PP_CUBIC 4

#define DF_PP_NATURAL 2
#define DF_PP_HERMITE 3
#define DF_PP_BESSEL 4
#define DF_PP_AKIMA 5

#define DF_NO_BC 0
#define DF_BC_NOT_A_KNOT 1
#define DF_BC_FREE_END 2
#define DF_BC_1ST_LEFT_DER 4
#define DF_BC_1ST_RIGHT_DER 8
#define DF_BC_PERIODIC 64

#define DF_NO_IC 0
#define DF_IC_1ST_DER 1

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
    elif sys.platform == "win32":
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
            locations.append(os.path.join(
                os.environ["HOME"], "intel/mkl/lib/intel64"))

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
    if rc != lib.DF_STATUS_OK:
        raise DfError(error_message(rc))


class DfTask:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.task = ffi.new("DFTaskPtr*")
        x_ptr = ffi.cast("double*", x.ctypes.data)
        y_ptr = ffi.cast("double*", y.ctypes.data)
        rc = lib.dfdNewTask1D(
            self.task, x.size, x_ptr, lib.DF_NON_UNIFORM_PARTITION, 1, y_ptr,
            lib.DF_NO_HINT)
        raise_if_error(rc)

    def editspline(self, s_type, bc_type=lib.DF_NO_BC, bc=None,
                   ic_type=lib.DF_NO_IC, ic=None):
        bc_ptr = ffi.NULL
        if bc is not None:
            self.bc = np.asarray(bc, dtype=np.float64)
            bc_ptr = ffi.cast("double*", self.bc.ctypes.data)

        ic_ptr = ffi.NULL
        if ic is not None:
            self.ic = np.asarray(ic, dtype=np.float64)
            ic_ptr = ffi.cast("double*", self.ic.ctypes.data)

        s_order = lib.DF_PP_CUBIC
        self.scoeff = np.empty((self.x.size-1)*s_order, dtype=self.x.dtype)
        scoeff_ptr = ffi.cast("double*", self.scoeff.ctypes.data)
        scoeffhint = lib.DF_NO_HINT

        rc = lib.dfdEditPPSpline1D(
            self.task[0], s_order, s_type, bc_type, bc_ptr, ic_type, ic_ptr,
            scoeff_ptr, scoeffhint)
        raise_if_error(rc)

    def editval(self, attr, val):
        rc = lib.dfiEditVal(self.task[0], attr, val)
        raise_if_error(rc)

    def editptr(self, attr, vals):
        if not hasattr(self, 'attr_ptr'):
            self.attr_ptr = {}
        vals = np.asarray(vals, dtype=np.float64)
        self.attr_ptr[attr] = vals           # keep alive
        rc = lib.dfdEditPtr(self.task[0], attr, ffi.cast(
            "double*", vals.ctypes.data))
        raise_if_error(rc)

    def construct(self):
        rc = lib.dfdConstruct1D(
            self.task[0], lib.DF_PP_SPLINE, lib.DF_METHOD_STD)
        raise_if_error(rc)

    def interpolate(self, site, ndorder=1):
        sitehint = lib.DF_NON_UNIFORM_PARTITION
        dorder = ffi.new("int[]", ndorder)
        dorder[ndorder-1] = 1
        # there is an error with Intel's example, datahint is a float array,
        # just happens that DF_NO_APRIORI_INFO=0
        # datahint = lib.DF_NO_APRIORI_INFO
        datahint = ffi.NULL
        r = np.empty_like(site)
        rhint = lib.DF_MATRIX_STORAGE_ROWS
        cell = ffi.NULL
        site_ptr = ffi.cast("double*", site.ctypes.data)
        r_ptr = ffi.cast("double*", r.ctypes.data)
        rc = lib.dfdInterpolate1D(
            self.task[0], lib.DF_INTERP, lib.DF_METHOD_PP, site.size, site_ptr,
            sitehint, ndorder, dorder, datahint, r_ptr, rhint, cell)
        raise_if_error(rc)
        return r

    def __del__(self):
        rc = lib.dfDeleteTask(self.task)
        raise_if_error(rc)


class CubicSpline:
    def __init__(self, x, y, bc_type='not-a-knot'):
        bc = None
        if bc_type == 'not-a-knot':
            bc_type = lib.DF_BC_NOT_A_KNOT
        elif bc_type == 'periodic':
            bc_type = lib.DF_BC_PERIODIC
        elif bc_type == 'natural':
            bc_type = lib.DF_BC_FREE_END
        elif bc_type == 'clamped':
            bc_type = lib.DF_BC_1ST_LEFT_DER | lib.DF_BC_1ST_RIGHT_DER
            bc = [0, 0]
        else:
            bc_type = lib.DF_NO_BC

        self.dftask = DfTask(x, y)
        self.dftask.editspline(lib.DF_PP_NATURAL, bc_type=bc_type, bc=bc)
        self.dftask.construct()

    def __call__(self, x, nu=0):
        return self.dftask.interpolate(x, ndorder=nu+1)


def pchipend(h1, h2, del1, del2):
    d = ((2*h1+h2)*del1 - h1*del2) / (h1+h2)
    if np.sign(d) != np.sign(del1):
        d = 0
    elif np.sign(del1) != np.sign(del2) and abs(d) > abs(3*del1):
        d = 3*del1
    return d


def pchipslopes(h, delta):
    # Numerical Computing with MATLAB chapter 3
    d = np.zeros(h.size+1)

    for k in range(1, d.size-1):
        if delta[k-1]*delta[k] > 0:    # same sign and neither zero
            w1 = 2*h[k] + h[k-1]
            w2 = h[k] + 2*h[k-1]
            d[k] = (w1+w2) / (w1/delta[k-1] + w2/delta[k])

    d[0] = pchipend(h[0], h[1], delta[0], delta[1])
    d[-1] = pchipend(h[-1], h[-2], delta[-1], delta[-2])
    return d


class PchipInterpolator:
    def __init__(self, x, y):
        h = np.diff(x)
        delta = np.diff(y) / h
        d = pchipslopes(h, delta)

        self.dftask = DfTask(x, y)
        bc_type = lib.DF_BC_1ST_LEFT_DER | lib.DF_BC_1ST_RIGHT_DER
        ic_type = lib.DF_IC_1ST_DER
        self.dftask.editspline(lib.DF_PP_HERMITE, bc_type=bc_type, bc=[
                               d[0], d[-1]], ic_type=ic_type, ic=d[1:-1])
        self.dftask.construct()

    def __call__(self, x, nu=0):
        return self.dftask.interpolate(x, ndorder=nu+1)


class Akima1DInterpolator:
    def __init__(self, x, y):
        # although mkl does support other boundary conditions for Akima,
        # scipy's Akima corresponds to no-boundary-condition
        self.dftask = DfTask(x, y)
        self.dftask.editspline(lib.DF_PP_AKIMA)
        self.dftask.construct()

    def __call__(self, x, nu=0):
        return self.dftask.interpolate(x, ndorder=nu+1)


def test_code(name):
    # 2.25 is chosen so that the ends have different derivatives
    freq = 2.25

    if name == 'periodic':
        freq = int(freq)

    # 100 Hz sampling
    x0 = 2*np.pi*freq*np.linspace(0, 1, 101)
    y0 = np.sin(x0)
    dy0 = np.cos(x0)

    if name == 'periodic':
        y0[-1] = y0[0]

    # 10 Hz sampling
    x1 = x0[::10].copy()
    y1 = y0[::10].copy()
    if name == 'akima':
        mkl_cs = Akima1DInterpolator(x1, y1)
        spi_cs = spi.Akima1DInterpolator(x1, y1)
    elif name == 'pchip':
        mkl_cs = PchipInterpolator(x1, y1)
        spi_cs = spi.PchipInterpolator(x1, y1)
    else:
        mkl_cs = CubicSpline(x1, y1, bc_type=name)
        spi_cs = spi.CubicSpline(x1, y1, bc_type=name)

    plt.plot(y0, label='T0')
    plt.plot(dy0, label='T1')
    plt.plot(spi_cs(x0), 'x', label='s0')
    plt.plot(spi_cs(x0, 1), 'x', label='s1')
    plt.plot(mkl_cs(x0), '.', label='m0')
    plt.plot(mkl_cs(x0, 1), '.', label='m1')

    plt.title(name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.interpolate as spi

    for name in ['not-a-knot', 'natural', 'periodic', 'clamped', 'akima',
                 'pchip']:
        test_code(name)
