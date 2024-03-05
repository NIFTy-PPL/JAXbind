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
  - name: Martin Reinecke (TODO check orcid)
    orcid: 0000-0001-6966-5658
    equal-contrib: true
    affiliation: "1"
  - name: Gordian Edenhofer
    equal-contrib: true
    orcid: 0000-0003-3122-4894
    affiliation: "1, 2"  # Multiple affiliations must be quoted
affiliations:
  - name: Max Planck Institute for Astrophysics, Karl-Schwarzschild-Straße 1, 85748 Garching bei München, Germany
    index: 1
  - name: Ludwig Maximilian University of Munich, Geschwister-Scholl-Platz 1, 80539 München, Germany
    index: 2
  - name: Technical University of Munich, Arcisstr. 21, 80333 München, Germany
    index: 3
date: 22 February 2024
bibliography: paper.bib
---

# Summary

* USP: easy-of-use and feature-completeness
TODO

# Statement of Need

The use of JAX [@Jax2018] is widespread in the natural sciences.
JAX's powerful transformation system is of especially high interest.
It enables retrieving arbitrary derivatives of functions, batch computations, and just-in-time compile code for additional performance.
The transformation system in JAX relies on all constituents of the computation being written in JAX.

A plethora of high-performance code is not written in JAX and thus not accessible from within JAX.
Rewriting these is often infeasible and/or inefficient.
Ideally, we would like to intermix existing high-performance code with JAX code.
However, connecting code to JAX requires knowledge of the internals of JAX and its C++ backend.

<!-- TODO: if we support JVPs, we can and should generalize this! -->
In this paper, we present `jax_op`, a package for bridging any function to JAX without in-depth knowledge of JAX's transformation system.
The interface is accessible from python with no C++ necessary.
The package is able to register any function, its partial derivatives and their transpose functions as a JAX native call, a so-called primitive.
Derivatives, compilation rules, and batching rules are automatically registered with JAX.

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We believe `jax_op` to be highly useful in scientific computing.
<!-- There are a lot of well-developed packages in JAX for, e.g., optimization and sampling that could be used once existing code is able to interface with JAX. -->
We intend to use this package to connect the Hartley transform and the spherical harmonic transform from ducc [@ducc0] to NIFTy [@Edenhofer2023NIFTyRE].
Furthermore, we intend to connect an image gridder implemented in C++ (TODO:cite resolve) for radio-astronomical data to JAX for use in radio-astronomy and strong-lensing astrophysics.
We envision many further applications within and outside of astrophysics, e.g., for highly specialized and well-optimized codes such as TODO.

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
To the best of our knowledge there exists no code for connecting generic functions to JAX.
The package that comes the closest is Enzyme-JAX [@Moses2024].
Enzyme-JAX allows one to differentiate a C++ function with Enzyme [@Moses2020; @Moses2021; @Moses2022] and connect it together with its derivative to JAX.
However, it enforces the use of Enzyme for deriving derivatives and does not allow for connecting arbitrary code to JAX.

# Automatic Differentiation
Automatic differentiation is a core feature of Jax and often a main reason for using Jax.
Therefore, it is essential that also custom Jax primitives registered by the user support automatic differentiation.
In the following we will outline which functions Jax and our `jax_op` package require to enable automatic differentiation for the custom primitive.

Assume we want to expose the function $f:X \mapsto Y$ as a Jax primitive.
Obviously, Jax needs an implementation of $f$ to evaluate the primitive.
Additionally to evaluating $f$, Jax can compute jacobian-vector products `jvp` and vector-jacobian products `vjp` of $f$, also referred to as forward and reverse mode differentiation.

The Jacobian-vector product in Jax is a function applying the Jacobian of $f$ at a position $x$ to tangent vectors.
Mathematically, this operation is called pushforward of $f$ and can be denoted as $\partial f(x): T_x X \mapsto T_{f(x)} Y$, with $T_x X$ and $T_{f(x)} Y$ being the tangent spaces of $X$ and $Y$ at the positions $x$ and $f(x)$.
As the implementation of $f$ is not Jax native, Jax cannot automatically compute the `jvp`.
Instead, an implementation of the pushforward has to be provided, which `jax_op` will register as the `jvp` of jax primitive of $f$.
More precisely, `jax_op` requires a function taking as an input a position $x\in X$ and tanged $v \in T_x$ and computing the corresponding cotangent $\partial f(x)(v)$.

The vector-Jacobian product in Jax is the linear transposed of the Jacobian-vector product.
In mathematical nomenclature this is the pullback $(\partial f(x))^{T}: T_{f(x)}Y \mapsto T_x X$ of $f$.
In analogy to `jvp`, the user has to implement this function as Jax cannot automatically construct it.
Specifically, the user has to provide a function computing the tangent vector $(\partial f(x))^{T}(c)$ for a given position $x \in X$ and cotangent $c \in T_{f(x)}Y$.

Jax supports higher order derivatives and thus can differentiate a `jvp` or `vjp` with respect to the position at which the Jacobian was taken.
Similar to first derivatives, there is no way for Jax to automatically compute higher derivatives of a general function $f$ that is not natively implemented in Jax.
Higher order derivatives would again need to be provided by the user.
Nevertheless, for many algorithms, first derivatives are sufficient, and higher order derivatives are often not implemented by the high-performance codes, which could be exposed to Jax.
Therefore, the current interface of `jax_op` is, for simplicity, restricted to first derivatives.
In the future, the interface could be expanded if specific use cases require higher order derivatives.

In scientific computing, linear functions such as spherical harmonic transformations are a frequently encountered special type.
If the function $f$ is linear, differentiation becomes much simpler.
Specifically for a linear $f$, the jacobian or the pushforward/ `jvp` of $f$ are identical to $f$ itself and are independent from the position at which they were computed.
Analogously, the pullback/ `vjp` becomes independent of the initial position and is just given by the linear transposed of $f$.
Also, all higher order derivatives can be expressed in terms of $f$ and its transpose.
To make use of these simplifications, `jax_op` provides a special interface, supporting higher order derivatives for linear functions, which only requires an implementation of the function and linear transposed function.

# Code Examples


# Acknowledgements

We would like to thank Dan Foreman-Mackey for his detailed guide (https://dfm.io/posts/extending-jax/) on connecting C++ code to JAX.
Jakob Roth acknowledges financial support by the German Federal Ministry of Education and Research (BMBF) under grant 05A23WO1 (Verbundprojekt D-MeerKAT III).
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