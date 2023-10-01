def trapezoidal_rule(func, a, b, N):
    h = (b - a) / N
    s = 0.5 * (func(a) + func(b))
    for i in range(1, N):
        s += func(a + i * h)
    return h * s

f = lambda x: x**4 - 2*x + 1

I1 = trapezoidal_rule(f, 0, 2, 10)
I2 = trapezoidal_rule(f, 0, 2, 20)

e2 = abs((1/3) * (I2 - I1))
true_error = abs(I2 - 4.4)
