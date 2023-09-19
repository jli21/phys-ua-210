import math

def quadratic_standard(a, b, c):
    discriminant = b**2 - 4*a*c
    
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    return root1, root2

def quadratic_alternative(a, b, c):
    discriminant = b**2 - 4*a*c

    root1 = (2*c) / (-b + math.sqrt(discriminant))
    root2 = (2*c) / (-b - math.sqrt(discriminant))
    return root1, root2

a = 0.001
b = 1000
c = 0.001

print("Using Standard Formula:")
standard_roots = quadratic_standard(a, b, c)

print("Root 1:", standard_roots[0])
print("Root 2:", standard_roots[1])

print("\nUsing Alternative Formula:")
alternative_roots = quadratic_alternative(a, b, c)
print("Root 1:", alternative_roots[0])
print("Root 2:", alternative_roots[1])

def accurate_quadratic_roots(a, b, c):
    D = math.sqrt(b**2 - 4*a*c)
    
    term1 = -b + D
    term2 = -b - D
    
    if abs(term1) < 1e-7:  
        root1 = 2*c / term2
    else:
        root1 = term1 / (2*a)
    
    if abs(term2) < 1e-7:  
        root2 = 2*c / term1
    else:
        root2 = term2 / (2*a)
        
    return root1, root2

print("\nUsing Accurate Quadratic Roots:")
root1, root2 = accurate_quadratic_roots(a, b, c)
print(f"Root 1: {root1}")
print(f"Root 2: {root2}")
