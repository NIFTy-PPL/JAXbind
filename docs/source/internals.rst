Internals of JAXbind
====================

This document outlines some important implementation choices underlying JAXbind.

JAXbind is a mixture of C++ and Python code that registers a `custom primitive <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`_ with JAX.
The primitive is registered together with its Jacobian-vector product and the vector-Jacobian product.
In the language of JAX, the transpose of the vector-Jacobian product is registered.
Furthermore, a batching rule and a lowering for just-in-time compilation is registered.
The registration with JAX of these functions is done in Python.

As of now, higher order derivates or recursive derivates are not supported in `JAXbind` except for linear functions.
This is because `JAXbind` tries to register all functions at once and would need to keep track of a large array of functions for arbitrary high derivates.
In the future, we might allow for derivates to be JAX primitives themselves and allow for recursive derivates.
Please reach out to us via the issue tracker if this is of interest to you!

While the registration with JAX is done in Python, the underlying call that is registered with JAX is in C++.
After just-in-time compilation, the call looks like any other C++ function to JAX.
Internally, the C++ function calls back to Python and calls the user-specified function.

The call to Python from C++ is always necessary as to expose a Python interface.
Ideally, the user-specified functions very quickly goes back to C, C++, Julia, rust, or any other high-performance language.
Usually, this is the case when, e.g., wrapping existing high performance scientific code for JAX.
The mere call to Python requires that the C++ call acquires Python's global interpreter lock (GIL).
The code in C++ is optimized to do as little as possible while holding the GIL.
Ideally, the code that is called in Python then very quickly releases the GIL once it went back to C, C++, etc.
Please file an issue if you think the GIL is slowing down your program!

The call from C++ to Python is very similar to the external callback interface in JAX.
Compared to the interface in JAX, `JAXbind` allows for both a custom Jacobian-vector product and a custom vector-Jacobian product while JAX only allows for one at a time.
`JAXbind` is able to support both as it registers both of them directly with the primary function while JAX relies on its automatic differentiation engine to derive derivates also for external callbacks and unfortunately only one of the two Jacobian products can be customized in JAX at a time.
Eventually, this limitation will hopefully be resolved with JAX's envisioned custom transposition backend which would allow defining a custom transpose, e.g., for the Jacobian-vector product, effectively allowing both a custom Jacobian-vector product and a custom vector-Jacobian product.

`JAXbind` should be thought of as an escape hatch that bridges tools to JAX that are otherwise difficult or too timeconsuming to code natively in JAX.
Currently, `JAXbind` limits itself to CPU code.
This constraint could be easily lifted once the need arises to bridge GPU code to JAX.
Please file an issue if this interests you!

An alternative to consider when thinking about writing native GPU code and bridging it to JAX is Pallas.
Pallas is JAX's `Triton <https://triton-lang.org>`_ frontend.
It natively interacts with JAX with no C++ in between.
Pallas relies on JAX's automatic differentiation engine and, as described above, might be limited in the kind of Jacobian products it can compute.
