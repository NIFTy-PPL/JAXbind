JAXbind Manual
==============

Welcome to the JAXbind documentation!

The existing interface in JAX for connecting fully differentiable custom code requires deep knowledge of JAX and its C++ backend.
The aim of `JAXbind` is to drastically lower the burden of connecting custom functions implemented in other programming languages to JAX.
Specifically, `JAXbind` provides an easy-to-use Python interface for defining custom, so-called JAX primitives.
Via `JAXbind`, any function callable from Python can be exposed as a JAX primitive.
`JAXbind` allows to interface the JAX function transformation engine with custom derivatives and batching rules, enabling all JAX transformations for the custom primitive.
In contrast, the JAX built-in `external callback interface <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_ also has a Python endpoint but cannot be fully integrated into the JAX transformation engine, as only the
`Jacobian-vector product <https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html>`_ or the `vector-Jacobian product <https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_vjp.html>`_ can be added but not both.

.. toctree::
   :maxdepth: 1

   Demos <demos/index>
   Jaxducc0 <contrib/index>
   Internals <internals>
   API reference <mod/jaxbind>
