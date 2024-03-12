# JAXbind: Easy bindings to JAX

Found a bug? [github.com/NIFTy-PPL/JAXbind/issues](https://github.com/NIFTy-PPL/JAXbind/issues)
 | Need help? [github.com/NIFTy-PPL/JAXbind/discussions](https://github.com/NIFTy-PPL/JAXbind/discussions)

## Summary

The existing interface in JAX for connecting custom code requires deep knowledge of JAX and its C++ backend.
The aim of `JAXbind` is to drastically lower the burden of connecting custom functions implemented in other programming languages to JAX.
Specifically, `JAXbind` provides an easy-to-use Python interface for defining custom, so-called JAX primitives supporting any JAX transformations.

### Automatic Differentiation and Code Example

Automatic differentiation is a core feature of JAX and often one of the main reasons for using it.
Thus, it is essential that custom functions registered with JAX support automatic differentiation.
In the following, we will outline which functions our package respectively JAX requires to enable automatic differentiation.
For simplicity, we assume that we want to connect the nonlinear function $f(x_1,x_2) = x_1x_2^2$ to JAX.
The `JAXbind` package expects the Python function for $f$ to take three positional arguments.
The first argument, `out`, is a `tuple` into which the function results are written.
The second argument is also a `tuple` containing the input to the function, in our case, $x_1$ and $x_2$.
Via `kwargs_dump`, potential keyword arguments given to the later registered Jax primitive can be forwarded to `f` in serialized form.

```python
import jaxbind

def f(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    x1, x2 = args
    out[0][()] = x1 * x2**2
```

JAX's automatic differentiation engine can compute the Jacobian-vector product `jvp` and vector-Jacobian product `vjp` of JAX primitives.
The Jacobian-vector product in JAX is a function applying the Jacobian of $f$ at a position $x$ to a tangent vector.
In mathematical nomenclature this operation is called the pushforward of $f$ and can be denoted as $\partial f(x): T_x X \mapsto T_{f(x)} Y$, with $T_x X$ and $T_{f(x)} Y$ being the tangent spaces of $X$ and $Y$ at the positions $x$ and $f(x)$.
As the implementation of $f$ is not JAX native, JAX cannot automatically compute the `jvp`.
Instead, an implementation of the pushforward has to be provided, which `JAXbind` will register as the `jvp` of the JAX primitive of $f$.
For our example, this Jacobian-vector-product function is given by $\partial f(x_1,x_2)(dx_1,dx_2) = x_2^2dx_1 + 2x_1x_2dx_2$.

```python
def f_jvp(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    x1, x2, dx1, dx2 = args
    out[0][()] = x2**2 * dx1 + 2 * x1 * x2 * dx2
```

The vector-Jacobian product `vjp` in JAX is the linear transpose of the Jacobian-vector product.
In mathematical nomenclature this is the pullback $(\partial f(x))^{T}: T_{f(x)}Y \mapsto T_x X$ of $f$.
Analogously to the `jvp`, the user has to implement this function as JAX cannot automatically construct it.
For our example function, the vector-Jacobian product is $(\partial f(x_1,x_2))^{T}(dy) = (x_2^2dy, 2x_1x_2dy)$.

```python
def f_vjp(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    x1, x2, dy = args
    out[0][()] = x2**2 * dy
    out[1][()] = 2 * x1 * x2 * dy
```

To just-in-time compile the function, JAX needs to abstractly evaluate the code, i.e. it needs to be able to know the shape and dtype of the output of the custom function given only the shape and dtype of the input.
We have to provide these abstract evaluation functions returning the output shape and dtype given an input shape and dtype for `f` as well as for the `vjp` application.
The output shape of the `jvp` is identical to the output shape of `f` itself and does not need to be specified again.
Due to the internals of JAX the abstract evaluation functions take normal keyword arguments and not serialized keyword arguments.

```python
def f_abstract(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype),)

def f_abstract_T(*args, **kwargs):
    return (
        (args[0].shape, args[0].dtype),
        (args[0].shape, args[0].dtype),
    )
```

We have now defined all ingredients necessary to register a JAX primitive for our function $f$ using the `JAXbind` package.

```python
f_jax = jaxbind.get_nonlinear_call(
    f, (f_jvp, f_vjp), f_abstract, f_abstract_T
)
```

`f_jax` is a JAX primitive registered via the `JAXbind` package supporting all JAX transformations.
We can now compute the `jvp` and `vjp` of the new JAX primitive and even jit-compile and batch it.

```python
import jax
import jax.numpy as jnp

inp = (jnp.full((4,3), 4.), jnp.full((4,3), 2.))
tan = (jnp.full((4,3), 1.), jnp.full((4,3), 1.))
res, res_tan = jax.jvp(f_jax, inp, tan)

cotan = (jnp.full((4,3), 6.),)
res, f_vjp = jax.vjp(f_jax, *inp)
res_cotan = f_vjp(cotan)

f_jax_jit = jax.jit(f_jax)
res = f_jax_jit(*inp)
```

### Higher Order Derivatives and Linear Functions

JAX supports higher order derivatives and can differentiate a `jvp` or `vjp` with respect to the position at which the Jacobian was taken.
Similar to first derivatives, JAX can not automatically compute higher derivatives of a general function $f$ that is not natively implemented in JAX.
Higher order derivatives would again need to be provided by the user.
For many algorithms, first derivatives are sufficient, and higher order derivatives are often not implemented by the high-performance codes.
Therefore, the current interface of `JAXbind` is, for simplicity, restricted to first derivatives.
In the future, the interface could be easily expanded if specific use cases require higher order derivatives.

In scientific computing, linear functions such as, e.g., spherical harmonic transforms are widespread.
If the function $f$ is linear, differentiation becomes trivial.
Specifically for a linear function $f$, the pushforward respectively the `jvp` of $f$ is identical to $f$ itself and independent of the position at which it is computed.
Expressed in formulas, $\partial f(x)(dx) = f(dx)$ if $f$ is linear in $x$.
Analogously, the pullback respectively the `vjp` becomes independent of the initial position and is given by the linear transpose of $f$, thus $(\partial f(x))^{T}(dy) = f^T(dy)$.
Also, all higher order derivatives can be expressed in terms of $f$ and its transpose.
To make use of these simplifications, `JAXbind` provides a special interface for linear functions, supporting higher order derivatives, only requiring an implementation of the function and its transpose.

### Demos

Additional demos can be found in the demos folder.
Specifically, there is a basic demo [01_linear_function.py](https://github.com/NIFTy-PPL/JAXbind/blob/multi_arg/demos/01_linear_function.py) showcasing the interface for linear functions and custom batching rules.
[02_multilinear_function.py](https://github.com/NIFTy-PPL/JAXbind/blob/multi_arg/demos/02_multilinear_function.py) binds a multi-linear function as a JAX primitive.
Finally, [03_nonlinear_function.py](https://github.com/NIFTy-PPL/JAXbind/blob/multi_arg/demos/03_nonlinear_function.py) demonstrates the interface for non-linear functions and shows how to deal with fixed arguments, which cannot be differentiated.


## Platforms

Currently, `JAXbind` only has CPU but no GPU support.
With some expertise on Python bindings for GPU kernels adding GPU support should be fairly simple.
The Interfacing with the JAX automatic differentiation engine is identical for CPU and GPU.
Contributions are welcome!

## Requirements

- [Python >= 3.8](https://www.python.org/)
- only when compiling from source: [pybind11](https://github.com/pybind/pybind11)
- only when compiling from source: a C++17-capable compiler, e.g.
  - `g++` 7 or later
  - `clang++`
  - MSVC 2019 or later
  - Intel `icpx` (oneAPI compiler series). (Note that the older `icpc` compilers
    are not supported.)

## Installation

FIXME: PyPi Installation!

To install JAXbind from source clone the repository and install JAXbind via pip.

```
git clone https://github.com/NIFTy-PPL/jaxbind.git
cd jaxbind
pip install --user .
```

## TODOs

* Paper
  * final editing
* PiPy release
* README

## Licensing terms

All source code in this package is released under the 2-clause BSD license.
All of JAXbind is distributed *without any warranty*.
