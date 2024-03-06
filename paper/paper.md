---
title: '`jax_op`: Use any function in JAX'
tags:
  - Python
  - Machine Learning
  - High Performance Computing
authors:
  - name: Jakob Roth
    orcid: 0000-0002-8873-8215
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Martin Reinecke
    orcid:
    equal-contrib: true
    affiliation: "1"
  - name: Gordian Edenhofer
    equal-contrib: true
    orcid: 0000-0003-3122-4894
    affiliation: "1, 2"  # Multiple affiliations must be quoted
affiliations:
  - name: Max Planck Institute for Astrophysics, Karl-Schwarzschild-Str. 1, 85748 Garching, Germany
    index: 1
  - name: Ludwig Maximilian University of Munich, Geschwister-Scholl-Platz 1, 80539 Munich, Germany
    index: 2
  - name: Technical University of Munich, Boltzmannstr. 3, 85748 Garching, Germany
    index: 3
date: 22 February 2024
bibliography: paper.bib
---

# Summary

JAX is widely used in machine learning and scientific computing.
Scientific computing relies on existing high-performance code which we would ideally like to use in JAX.
Reimplementing the existing code in JAX is often impractical and the existing interface in JAX for connecting custom code requires deep knowledge of JAX and its C++ backend.
The aim of `jax_op` is to drastically lower the burden of connecting custom functions implemented in other programming languages to JAX.
Specifically, `jax_op` provides an easy-to-use Python interface for defining custom, so-called JAX primitives supporting any JAX transformations.


# Statement of Need

The use of JAX [@Jax2018] is widespread in the natural sciences.
JAX's powerful transformation system is of especially high interest.
It enables retrieving arbitrary derivatives of functions, batch computations, and just-in-time code compilation for additional performance.
Its transformation system relies on all constituents of the computation being written in JAX.

A plethora of high-performance code is not written in JAX and thus not accessible from within JAX.
Rewriting these is often infeasible and/or inefficient.
Ideally, we would like to intermix existing high-performance code with JAX code.
However, connecting code to JAX requires knowledge of the internals of JAX and its C++ backend.

In this paper, we present `jax_op`, a package for bridging any function to JAX without in-depth knowledge of JAX's transformation system.
The interface is accessible from Python without requiring any development in C++.
The package is able to register any function, its partial derivatives and their transpose functions as a JAX native call, a so-called primitive.
Derivatives, compilation rules, and batching rules are automatically registered with JAX.

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We believe `jax_op` to be highly useful in scientific computing.
<!-- There are a lot of well-developed packages in JAX for, e.g., optimization and sampling that could be used once existing code is able to interface with JAX. -->
We intend to use this package to connect the Hartley transform and the spherical harmonic transform from ducc [@ducc0] to NIFTy [@Edenhofer2023NIFTyRE] as well as the radio interferometry response from ducc with \texttt{resolve} [@Resolve2024] for radio astronomy.
Furthermore, we intend to connect the non-uniform FFT from ducc with JAX for applications in strong-lensing astrophysics.
We envision many further applications within and outside of astrophysics.

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
To the best of our knowledge no other code currently exists for connecting generic functions to JAX.
The package that comes the closest is Enzyme-JAX [@Moses2024].
Enzyme-JAX allows one to differentiate a C++ function with Enzyme [@Moses2020; @Moses2021; @Moses2022] and connect it together with its derivative to JAX.
However, it enforces the use of Enzyme for deriving derivatives and does not allow for connecting arbitrary code to JAX.

# Automatic Differentiation and Code Example

Automatic differentiation is a core feature of JAX and often one of the main reasons for using it.
Thus, it is essential that custom functions registered with JAX support automatic differentiation.
In the following, we will outline which functions our package respectively JAX requires to enable automatic differentiation.
For simplicity, we assume that we want to connect the nonlinear function $f(x_1,x_2) = x_1x_2^2$ to JAX.
The `jax_op` package expects the Python function for $f$ to take three positional arguments.
The first argument, `out`, is a `tuple` into which the function results are written.
The second argument is also a `tuple` containing the input to the function, in our case, $x_1$ and $x_2$.
Via `kwargs_dump`, potential keyword arguments given to the later registered Jax primitive can be forwarded to `f` in serialized form.

```python
import jax_linop #FIXME

def f(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x1, x2 = args
    out[0][()] = x1 * x2**2
```

JAX's automatic differentiation engine can compute the Jacobian-vector product `jvp` and vector-Jacobian product `vjp` of JAX primitives.
The Jacobian-vector product in JAX is a function applying the Jacobian of $f$ at a position $x$ to a tangent vector.
In mathematical nomenclature this operation is called the pushforward of $f$ and can be denoted as $\partial f(x): T_x X \mapsto T_{f(x)} Y$, with $T_x X$ and $T_{f(x)} Y$ being the tangent spaces of $X$ and $Y$ at the positions $x$ and $f(x)$.
As the implementation of $f$ is not JAX native, JAX cannot automatically compute the `jvp`.
Instead, an implementation of the pushforward has to be provided, which `jax_op` will register as the `jvp` of the JAX primitive of $f$.
For our example, this Jacobian-vector-product function is given by $\partial f(x_1,x_2)(dx_1,dx_2) = x_2^2dx_1 + 2x_1x_2dx_2$.

```python
def f_jvp(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x1, x2, dx1, dx2 = args
    out[0][()] = x2**2 * dx1 + 2 * x1 * x2 * dx2
```

The vector-Jacobian product `vjp` in JAX is the linear transpose of the Jacobian-vector product.
In mathematical nomenclature this is the pullback $(\partial f(x))^{T}: T_{f(x)}Y \mapsto T_x X$ of $f$.
Analogously to the `jvp`, the user has to implement this function as JAX cannot automatically construct it.
For our example function, the vector-Jacobian product is $(\partial f(x_1,x_2))^{T}(dy) = (x_2^2dy, 2x_1x_2dy)$.

```python
def f_vjp(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x1, x2, dy = args
    out[0][()] = x2**2 * dy
    out[1][()] = 2 * x1 * x2 * dy
```

To just-in-time compile the function, JAX needs to abstractly evaluate the code, i.e. it needs to be able to know the shape and dtype of the output of the custom function given only the shape and dtype of the input.
We have to provide these abstract evaluation functions returning the output shape and dtype given an input shape and dtype for `f` as well as for the `vjp` application.
The output shape of the `jvp` is identical to the output shape of `f` itself and does not need to be specified again.
<!-- Should we point out specifically that the abstract functions take "traditional" args and kwargs? -->

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

We have now defined all ingredients necessary to register a JAX primitive for our function $f$ using the `jax_op` package.

```python
f_jax = jax_linop.get_nonlinear_call(
    f, (f_jvp, f_vjp), f_abstract, f_abstract_T
)
```

`f_jax` is a JAX primitive registered via the `jax_op` package supporting all JAX transformations.
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

# Higher Order Derivatives and Linear Functions

JAX supports higher order derivatives and can differentiate a `jvp` or `vjp` with respect to the position at which the Jacobian was taken.
Similar to first derivatives, JAX can not automatically compute higher derivatives of a general function $f$ that is not natively implemented in JAX.
Higher order derivatives would again need to be provided by the user.
For many algorithms, first derivatives are sufficient, and higher order derivatives are often not implemented by the high-performance codes.
Therefore, the current interface of `jax_op` is, for simplicity, restricted to first derivatives.
In the future, the interface could be easily expanded if specific use cases require higher order derivatives.

In scientific computing, linear functions such as, e.g., spherical harmonic transforms are widespread.
If the function $f$ is linear, differentiation becomes trivial.
Specifically for a linear function $f$, the pushforward respectively the `jvp` of $f$ is identical to $f$ itself and independent of the position at which it is computed.
Expressed in formulas, $\partial f(x)(dx) = f(dx)$ if $f$ is linear in $x$.
Analogously, the pullback respectively the `vjp` becomes independent of the initial position and is given by the linear transpose of $f$, thus $(\partial f(x))^{T}(dy) = f^T(dy)$.
Also, all higher order derivatives can be expressed in terms of $f$ and its transpose.
To make use of these simplifications, `jax_op` provides a special interface for linear functions, supporting higher order derivatives, only requiring an implementation of the function and its transpose.

# Acknowledgements

We would like to thank Dan Foreman-Mackey for his detailed guide (https://dfm.io/posts/extending-jax/) on connecting C++ code to JAX.
Jakob Roth acknowledges financial support from the German Federal Ministry of Education and Research (BMBF) under grant 05A23WO1 (Verbundprojekt D-MeerKAT III).
Gordian Edenhofer acknowledges support from the German Academic Scholarship Foundation in the form of a PhD scholarship ("Promotionsstipendium der Studienstiftung des Deutschen Volkes").

# References

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->
