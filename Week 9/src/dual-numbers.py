"""
Dual Numbers and Automatic Differentiation

This is a demo notebook for the Masterclass "Dual Numbers and Automatic Differentiation" at University College London (UCL) in the academic year 2024 / 2025.

Author: [Martin Benning](mailto:martin.benning@ucl.ac.uk)

Date: 30.03.2020

Last updated: 05.12.2025

First we load the necessary libraries.
"""

from __future__ import annotations
import math

# ## Dual numbers and automatic differentiation
# 
# In this notebook we want to implement the forward mode of automatic differentiation with the help of dual numbers. We first implement a class **Dual** with the constructor **__init__**, the functions **__add__**, **__radd__**, **__sub__**, **__rsub__**, **__mul__**, **__rmul__**, **__truediv__**, **__rtruediv__**, **__neg__** and **__pow__**. As the names suggest, those  functions and properties implement basic arithmetic operations for Dual numbers:
# 
# __init__ : constructor that initialises an object of class **Dual**. Each object represents a dual number $a + \varepsilon \, b$ with real component $a$ (*self.real*) and dual component $b$ (*self.dual*).
# 
# __add__ : adds an argument _argument_ to the dual number, i.e. $a + \varepsilon \, b + \text{argument}$. 
# 
# __radd__ : adds the dual number to the argument _argument_, i.e. $\text{argument} + a + \varepsilon \, b$.
# 
# __sub__ : subtracts an argument _argument_ from the dual number. 
# 
# __rsub__ : subtracts the dual number from the argument _argument_.
# 
# __mul__ : multiplies the dual number with the argument _argument_.
# 
# __rmul__ : multiplies an argument _argument_ with the dual number. 
# 
# __truediv__ : divides the dual number by an argument _argument_.
# 
# __rtruediv__ : divides the argument _argument_ by the dual number.
# 
# __neg__ : returns the negative of the dual number $a + \varepsilon b$, i.e. $-a - \varepsilon b$.
# 
# __pow__ : takes the _power_-th power of the dual number.

class Dual:
    
    def __init__(self, real: float | int, dual: float | int):
        self.real = real
        self.dual = dual
        
    def __add__(self, argument: Dual | float | int) -> Dual:
        if isinstance(argument, Dual):
            return Dual(self.real + argument.real, self.dual + argument.dual)
        else:
            return Dual(self.real + argument, self.dual)
        
    __radd__ = __add__
    
    def __sub__(self, argument: Dual | float | int) -> Dual:
        if isinstance(argument, Dual):
            return Dual(self.real - argument.real, self.dual - argument.dual)
        else:
            return Dual(self.real - argument, self.dual)
        
    def __rsub__(self, argument: float | int) -> Dual:
        return Dual(argument, 0) - self
    
    def __mul__(self, argument: Dual | float | int) -> Dual:
        if isinstance(argument, Dual):
            return Dual(self.real * argument.real, self.real * argument.dual + \
                self.dual * argument.real)
        else:
            return Dual(self.real * argument, self.dual * argument.real)
        
    __rmul__ = __mul__
    
    def __truediv__(self, argument: Dual | float | int) -> Dual:
        if isinstance(argument, Dual):            
            if argument.real == 0:
                raise ZeroDivisionError('Real part of dual number is zero.')
            else:
                return Dual(self.real / argument.real, (self.dual * argument.real - \
                        self.real * argument.dual)/(argument.real ** 2))
        else:
            if argument == 0:
                raise ZeroDivisionError('Argument is zero.')
            else:
                return Dual(self.real / argument, self.dual)
        
    def __rtruediv__(self, argument: float | int) -> Dual:
        return Dual(argument, 0).__truediv__(self)
        
    def __neg__(self) -> Dual:
        return Dual(-self.real, -self.dual)
    
    def __pow__(self, power: float | int) -> Dual:
        if power < 1:
            raise ValueError('The power needs to be larger than one.')
        else:
            return Dual(self.real ** power, power * self.dual * (self.real ** \
                            (power - 1)))
    
    def __repr__(self) -> str:                    
        if self.dual == 0:
            representation = repr(self.real)    
        elif self.dual == 1:
            representation = repr(self.real) + ' + epsilon'
        elif self.dual == -1:
            representation = repr(self.real) + ' - epsilon'
        elif self.dual > 0:
            representation = repr(self.real) + ' + ' + repr(self.dual) + ' * epsilon'
        else:
            representation = repr(self.real) + ' - ' + repr(-self.dual) + ' * epsilon'
        return representation
    
    def __str__(self) -> str:
        if self.dual == 0:
            representation = str(self.real)    
        elif self.dual == 1:
            representation = str(self.real) + " + epsilon"
        elif self.dual == -1:
            representation = str(self.real) + " - epsilon"
        elif self.dual > 0:
            representation = str(self.real) + " + " + str(self.dual) + " * epsilon"
        else:
            representation = str(self.real) + " - " + str(-self.dual) + " * epsilon"
        return representation

# We can now construct dual numbers such as $a = 9 - \varepsilon$ and $b = -3 + 7\varepsilon$ and multiply them to obtain $c = a b$.

a = Dual(9, -1)
b = Dual(-3, 7)
c = a * b
print(a,'\n\n',b,'\n\n',c, sep='')

# We can also write more complicated expressions, such as the polynomial 
# 
# $$\frac{(3x^2 - 2x + 5)^2}{5x^2 - 7x^3 - 1}$$
# 
# on a dual number $x = 3 + \varepsilon$, with the following lines of code.

x = Dual(3, 1)
polynomial = (3*x**2 - 2*x + 5) ** 2 / (-7*x**3 + 5*x**2 - 1)
print(polynomial)

# Next, we implement the following functions that are acting on dual numbers of the form $a + \varepsilon \, b$:
#     
# **log** : $\log(a + \varepsilon \, b)$
# 
# **exp** : $\exp(a + \varepsilon \, b)$
# 
# **sin** : $\sin(a + \varepsilon \, b)$
# 
# **cos** : $\cos(a + \varepsilon \, b)$
# 
# **sigmoid** : $\frac{1}{1 + \exp(-(a + \varepsilon \, b))}$
# 
# **tanh** : $\tanh(a + \varepsilon \, b)$

def log(dual_number: Dual) -> Dual:
    return Dual(math.log(dual_number.real), dual_number.dual / dual_number.real)

def exp(dual_number: Dual) -> Dual:
    return Dual(math.exp(dual_number.real), dual_number.dual * math.exp(dual_number.real))
    
def sin(dual_number: Dual) -> Dual:
    return Dual(math.sin(dual_number.real), dual_number.dual * math.cos(dual_number.real))
    
def cos(dual_number: Dual) -> Dual:
    return Dual(math.cos(dual_number.real), - dual_number.dual * math.sin(dual_number.real))

def sigmoid(dual_number: Dual) -> Dual:
    return 1/(1 + exp(-dual_number))

def tanh(dual_number: Dual) -> Dual:
    return Dual(math.tanh(dual_number.real), dual_number.dual * (1 - (math.tanh( \
                dual_number.real) ** 2)))

# Next, we compare the dual part of the function
# 
# $$ f(x) = \frac{\sin(3x^2 - 2x)}{\exp(1 + \cos(-5x^2))} \, ,$$
# 
# for the dual number $x = \sqrt{2} + \varepsilon$, with the symbolic derivative of $f$ at $\sqrt{2}$.

x = Dual(math.sqrt(2), 1)
f = sin(3*(x ** 2) - 2*x)/exp(1 + cos(-5*(x ** 2)))

def symbolic_derivative_of_f(x):
    output = math.exp(-math.cos(5*(x ** 2)) - 1) * ((6*x - 2)*math.cos((2 - 3*x)*x) - \
                    10*x*math.sin((2 - 3*x)*x)*math.sin(5*(x ** 2)))
    return output

print(f.dual, symbolic_derivative_of_f(x.real))

# In the presentation, we look at the function
# 
# $$ f(w_1, \ldots, w_l) = \frac12 \left( \tanh\left(w_l \tanh\left( w_{l - 1} \tanh\left( \cdots w_2 \tanh( w_1 x ) \right) \cdots \right) \right) - y \right)^2 $$
# 
# We can obviously calculate the derivate with respect to an individual $w_j$ by hand by using the chain rule, but we can do this also automatically with the help of dual numbers. 

x = Dual(-3, 0)
y = Dual(100, 0)
w = [Dual(-1, 0), Dual(5, 0), Dual(3/2, 0), Dual(-1/3, 1), Dual(20, 0)]
f = 1/2*(tanh(w[4]*tanh(w[3]*tanh(w[2]*tanh(w[1]*tanh(w[0] * x))))) - y) ** 2
print(f.dual)