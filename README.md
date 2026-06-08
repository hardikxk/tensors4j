# NADAR

## The Aim for this project
**Nadar** is aimed to provide a high dimensional numerical and tensor operations SIMD library for researchers and enterprises alike for Java (similar to PyTorch, TensorFlow, Jax, etc. for Python).

The project is aimed to provide high performance, parallel + GPU execution for numerical computations using the new [FFM API](https://docs.oracle.com/en/java/javase/25/core/foreign-function-and-memory-api.html) introduced in [Project Panama](https://openjdk.org/projects/panama/).

Furthermore, the Nadar is on road to also integrate the work of [Project Babylon](https://openjdk.org/projects/babylon/) along with FFM API for GPU computation for higher dimensional Tensors (which are right now being computed directly on the CPU itself).

## Checkout a [quick demo of the library](https://sargunai.github.io/nadar/demo).

## GOALS (expected to be completed in short term):
- Refactor the Tensor API for a better DX (similar to that of other ML/DL frameworks). Abstract away everything!
- Work up on the TODOs
- Add GPU support using incubator features of [Project Babylon](https://openjdk.org/projects/babylon/)
- Improve the Vector API implementation (because the current implementation while works can be improved)
- Validate the Arena tradeoffs and how to leverage concurrency to perhaps improve performance on the CPU.
- Keep up the docs!

### Check the Milestones and issues.

### Requirements to Build and work with Source
- JDK 25
- Gradle

> Or you can just use IntelliJ and the project should work out of the box!