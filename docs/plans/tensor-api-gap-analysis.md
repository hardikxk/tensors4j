# Tensor.java — Missing Methods Gap Analysis

## Source: VectorOps (12 public methods)
| VectorOps method | Has Tensor wrapper? | Missing? |
|---|---|---|
| add(a, b, arena) | ✅ Tensor.add | — |
| sub(a, b, arena) | ❌ | **MISSING: Tensor.sub** |
| mul(a, b, arena) | ✅ Tensor.mul | — |
| scale(a, scalar, arena) | ✅ Tensor.scale | — |
| relu(a, arena) | ✅ Tensor.relu | — |
| addInPlace(a, b) | ❌ | **MISSING: Tensor.add_** |
| scaleInPlace(a, scalar) | ✅ Tensor.scale_ | — |
| reluInPlace(a) | ✅ Tensor.relu_() | — |
| sum(a) | ✅ Tensor.sum | — |
| mean(a) | ✅ Tensor.mean | — |
| dot(a, b) | ✅ Tensor.dot | — |

## Source: VectorMulOps (1 public method)
| VectorMulOps method | Has Tensor wrapper? | Missing? |
|---|---|---|
| matmul(a, b, c) | ✅ Tensor.matmul | — |

## Source: NDArray (public methods)
| NDArray method | Has Tensor wrapper? | Missing? |
|---|---|---|
| transpose() | ❌ | **MISSING: Tensor.transpose()** |
| transpose(int... axes) | ❌ | **MISSING: Tensor.transpose(int...)** |
| reshape(long...) | ❌ | **MISSING: Tensor.reshape(long...)** |
| slice(long) | ❌ | **MISSING: Tensor.slice(long)** |
| getFloat(indices) | ✅ Tensor.getFloat | — |
| setFloat(v, indices) | ✅ Tensor.setFloat | — |
| getFloatAtIndex(i) | ❌ | Documented as NDArray-only |
| setFloatAtIndex(v, i) | ❌ | Documented as NDArray-only |
| zeros/ones/fill | ✅ Tensor.zeros/ones/fill | — |
| close() | ✅ Tensor.close | — |
| toString() | ✅ Tensor.toString | — |
| isContiguous() | ❌ | **MISSING: Tensor.isContiguous()** |
| shape/stride/dtype fields | ✅ shape()/dtype() wrappers | — |

## Summary of 6 Missing Methods

1. **Tensor.sub(Tensor other)** — allocating subtraction (a - b)
2. **Tensor.add_(Tensor other)** — in-place addition (this += other)
3. **Tensor.transpose()** — 2D transpose (convenience, delegates to NDArray)
4. **Tensor.transpose(int... axes)** — N-dim transpose (convenience, delegates to NDArray)
5. **Tensor.reshape(long... newShape)** — reshape view (delegates to NDArray)
6. **Tensor.slice(long index)** — first-dim slice (delegates to NDArray)
7. **Tensor.isContiguous()** — contiguity check (delegates to NDArray)

## Note on relu_(Tensor other)
The two-arg relu_(Tensor other) at line 115 mutates `other` instead of `this`. This is either dead code that should be removed, or should be renamed. It's NOT a standard operation a user expects.
