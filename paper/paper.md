---
title: 'jax_linop: Use any linear operators in JAX (TODO)'
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

* Use of JAX spread in astrophysics and physics more broadly
* JAX is used for its transformation system allowing one to compute arbitrary derivatives of functions and batch computations
* Furthermore, JAX, can effortlessly just-in-time compile code for additional performance.

* However, a huge plethora of high-performance code is not written in JAX and rewriting it in JAX is either infeasible or inefficient
* Oftentimes, the performance critical code is linear in its input and retrieving derivatives is trivial
<!-- TODO: if we support JVPs, we can and should generalize this! -->
* Ideally, we would be able to continue to use the existing high-performance code and intermix it with JAX
* So far, it has been very involved to connect existing code to JAX.
* It required deep knowledge of the internal behavior of JAX and documentation was sparse

* In this paper, we provide a simple solution for bridging any linear function to JAX
* Interface is in plain python with no C++ necessary
* The package registers a given the function and its transpose function as a JAX native so-called primitive
* Derivatives and compilation rules are then automatically defined for linear-functions

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We believe this `jax_linop` to be highly useful in scientific computing.
* There are a lot of well-developed packages in JAX for optimization and sampling that can be used once existing functions are connected to JAX
* We intend to use this package to connect the Hartley transform and the spheric harmonic transform from ducc [@ducc0] to NIFTy [@Edenhofer2023NIFTyRE]
* We envision many future applications such as, e.g., connecting the astrophysical particle propagation code PICARD (TODO:cite) to JAX and the gridder for radio-astronomical data in @ducc0 to strong-lensing and radio-imaging codes in JAX

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
* Some projects already connected their calls to JAX, e.g., FINUFFT (TODO:cite), TODO.
* Furthermore, enzyme connected their AD backend for C++ to JAX with automatic derivates https://github.com/EnzymeAD/Enzyme-JAX
* Some guidance on connecting C++ code to JAX is given in https://dfm.io/posts/extending-jax/ but to the best of our knowledge, no generic code for connecting arbitrary codes to JAX

# Acknowledgements

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
