import numpy as np
from scipy.optimize import brent

def func(x):
    return (x - 0.3)**2 * np.exp(x)

def brent_method(f, xb):
    xtol = 1e-15
    ytol = 1e-8
    xb = np.array(xb, dtype="double")
    yb = np.array([f(xb[0]), f(xb[1])])
    # Ensure that the left point is higher than the right
    if not (yb[0] > yb[1]):
        xb = np.array([xb[1], xb[0]])
        yb = np.array([yb[1], yb[0]])
    c = xb[0]
    yc = yb[0]
    d = None
    usedbisec = True
    while True:
        if usedbisec or (yb[0] != yc and yb[1] != yc):
            # Use inverse quadratic interpolation
            s = (xb[0] * yb[1] * yc / ((yb[0] - yb[1]) * (yb[0] - yc)) +
                 xb[1] * yb[0] * yc / ((yb[1] - yb[0]) * (yb[1] - yc)) +
                 c * yb[0] * yb[1] / ((yc - yb[0]) * (yc - yb[1])))
        else:
            # Use secant method
            s = xb[1] - yb[1] * (xb[1] - xb[0]) / (yb[1] - yb[0])
        # Check for convergence
        if ((s - (3 * xb[0] + xb[1]) / 4) * (s - xb[1]) >= 0 or
            (usedbisec and np.abs(s - xb[1]) >= np.abs(xb[1] - c) / 2) or
            (not usedbisec and d is not None and np.abs(s - xb[1]) >= np.abs(c - d) / 2) or
            (usedbisec and np.abs(xb[1] - c) < np.abs(xtol)) or
            (not usedbisec and d is not None and np.abs(c - d) < np.abs(xtol))):
            s = (xb[0] + xb[1]) / 2
            usedbisec = True
        else:
            usedbisec = False
        ys = f(s)
        d = c
        c = xb[1]
        yc = yb[1]
        if ys < yb[1]:
            xb[0] = xb[1]
            yb[0] = yb[1]
            xb[1] = s
            yb[1] = ys
        else:
            if ys < yb[0]:
                xb[0] = s
                yb[0] = ys
        # Ensure the lower function value is always in yb[1]
        if yb[0] < yb[1]:
            xb = np.array([xb[1], xb[0]])
            yb = np.array([yb[1], yb[0]])
        if np.abs(xb[1] - xb[0]) < xtol or np.abs(yb[1]) < ytol:
            x = xb[1]
            y = yb[1]
            break
    return x, y

brent_result = brent_method(func, [0, 2])
scipy_brent_result = brent(func, brack=(0, 1), tol=1e-8, full_output=True)

comp = {
    "Manual Brent's Method": {
        "Result": brent_result[0],
        "Function Value": brent_result[1],
    },
    "Scipy Brent's Method": {
        "Result": scipy_brent_result[0],
        "Function Value": scipy_brent_result[1],
        "Iterations": scipy_brent_result[3]
    }
}

comp
