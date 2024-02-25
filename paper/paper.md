---
title: '`jax_linop`: Use any linear operators in JAX (TODO)'
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

A plethora of high-performance code is not written in JAX and thus not accesible from within JAX.
Rewriting these is often infeasible and/or inefficient.
Ideally, we would like to intermix existing high-performance code with new JAX code.
However, connecting code to JAX requires knowledge of the internals of JAX and its C++ backend.

<!-- TODO: if we support JVPs, we can and should generalize this! -->
In this paper, we present `jax_linop`, a package for bridging any linear function to JAX without in-depth knowledge of JAX's transformation system.
The interface is accessible from python with no C++ necessary.
The package is able to register any linear function and its transpose as a JAX native call, a so-called primitive.
Derivatives, compilation rules, and batching rules are automatically registered with JAX.

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We believe `jax_linop` to be highly useful in scientific computing.
<!-- There are a lot of well-developed packages in JAX for, e.g., optimization and sampling that could be used once existing code is able to interface with JAX. -->
We intend to use this package to connect the Hartley transform and the spherical harmonic transform from ducc [@ducc0] to NIFTy [@Edenhofer2023NIFTyRE].
Furthermore, we intend to connect an image gridder implemented in C++ (TODO:cite resolve) for radio-astronomical data to JAX for use in radio-astronomy and strong-lensing astrophysics.
We envision many further applications inside and outside of astrophysics, e.g., for highly specialized and well-optimized codes such as TODO.

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
To the best of our knowledge there exists no code for connecting generic functions to JAX.
The package that comes the closest is Enzyme-JAX [@Moses2024].
Enzyme-JAX allows one to differentiate a C++ function with Enzyme [@Moses2020; @Moses2021; @Moses2022] and connect it together with its derivative to JAX.
However, it enforces the use of Enzyme for deriving derivatives and does not allow for plugging in arbitrary codes into JAX.

# Acknowledgements

We would like to thank Dan Foreman-Mackey for his detailed guide (https://dfm.io/posts/extending-jax/) on connecting C++ code to JAX.
Jakob Roth acknowledges financial support by the German Federal Ministry of Education and Research (BMBF) under grant 05A20W01 (Verbundprojekt D-MeerKAT).
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
