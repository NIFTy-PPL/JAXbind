JAXbind Manual
==============

Welcome to the JAXbind documentation!

The existing interface in JAX for connecting custom code requires deep knowledge of JAX and its C++ backend.
The aim of `JAXbind` is to drastically lower the burden of connecting custom functions implemented in other programming languages to JAX.
Specifically, `JAXbind` provides an easy-to-use Python interface for defining custom, so-called JAX primitives.
Thus, via `JAXbind`, any function callable from Python can be exposed as a JAX primitive.
Furthermore, `JAXbind` allows to interface the JAX function transformation engine with custom derivatives and batching rules, enabling all JAX transformations for the custom primitive.
In contrast, the JAX built-in `external callback <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_ also has a Python interface but cannot be fully integrated into the JAX transformation engine, as either the
`Jacobian-vector product <https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html>`_ or the `vector-Jacobian product <https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_vjp.html>`_ can be added.

.. toctree::
   :maxdepth: 1

   Demos <demos/index>
   Internals <internals>
   API reference <mod/jaxbind>
