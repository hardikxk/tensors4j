package io.tensors4j;

import java.lang.foreign.Arena;
import java.util.Arrays;

// TODO: Improve the docstring to mention all the methods provided by the Tensor class. (also the other docstrings)
// TODO: There can be a better way to initialize the arena in a different function and call that for every instantiation operation.

/**
 * Public API facade for tensors.
 * This API is the wrapper for all the internal mechanisms.
 * The design philosophy of typical idiomatic ML/Numeric computation APIs should be followed while developing this.
 */
public class Tensor implements AutoCloseable {

    private final NDArray array;
    private final Arena arena;

    // TODO: Allow Tensor reallocation from exising tensors
    private Tensor(Tensor t, Arena arena) {
        this.array = t.array;
        this.arena = arena;
    }

    private Tensor(NDArray array, Arena arena) {
        this.array = array;
        this.arena = arena;
    }

    /**
     * Creates a tensor of zeros with the given dtype and shape.
     */
    public static Tensor zeros(DType dtype, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.zeros(dtype, arena, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Creates a tensor of ones with the given dtype and shape.
     */
    public static Tensor ones(DType dtype, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.ones(dtype, arena, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Creates a tensor filled with {@code value} with the given dtype and shape.
     */
    public static Tensor fill(DType dtype, float value, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.fill(dtype, arena, value, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Allocating addition. Returns a new Tensor where each element is
     * the sum of the corresponding elements of {@code this} and {@code other}.
     * Both tensors must have the same shape.
     */
    public Tensor add(Tensor other) {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.add(this.array, other.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Allocating subtraction. Returns a new Tensor where each element if the difference of the corresponding elements of {@code this} and {@code other} both tensors must have the same shape.
     */
    public Tensor sub(Tensor other) {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.sub(this.array, other.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * In-place addition. Adds {@code other} to {@code this} element-wise.
     * mutates this tensor directly — no allocation.
     * NOTE :: both tensors must have the same shape.
     */
    public void add_(Tensor other) {
        VectorOps.addInPlace(this.array, other.ndarray());
    }

    /**
     * Returns a defensive copy of the shape array.
     * @return Tensor (consisting of the sum of the values).
     */
    public long[] shape() {
        return Arrays.copyOf(array.shape, array.shape.length);
    }

    public DType dtype() {
        return array.dtype;
    }

    /**
     * Number of dimensions.
     */
    public int ndims() {
        return array.shape.length;
    }

    /**
     * Total number of elements.
     */
    public long size() {
        return NDArray.elementCount(array.shape);
    }

    /**
     * Returns whether this tensor's backing array is contiguous in row-major order.
     */
    public boolean isContiguous() {
        return array.isContiguous();
    }

    /**
     * Returns a transposed view of this tensor (2D only).
     * Swaps axes 0 and 1. No data is copied.
     * @return a new Tensor that is a transposed view of this one
     * @throws UnsupportedOperationException if this tensor is not 2D
     */
    public Tensor transpose() {
        return new Tensor(array.transpose(), arena);
    }

    /**
     * Returns a transposed view of this tensor with axes permuted.
     * No data is copied. The returned tensor shares the same underlying memory.
     * @param axes a permutation of {0, 1, ..., n-1} specifying the new order of axes
     * @return a new Tensor that is a transposed view of this one
     * @throws IllegalArgumentException if axes is not a valid permutation
     */
    public Tensor transpose(int... axes) {
        return new Tensor(array.transpose(axes), arena);
    }

    /**
     * Returns a *view* of this tensor with a different logical shape that is specified.
     * *The tensor must be contiguous*.
     * @param newShape the desired shape (element count must match)
     * @return a new Tensor view with the given shape
     * @throws UnsupportedOperationException if this tensor is not contiguous
     * @throws IllegalArgumentException if the element counts do not match
     */
    public Tensor reshape(long... newShape) {
        return new Tensor(array.reshape(newShape), arena);
    }

    /**
     * Returns a view of the first dimension at the given index.
     * No data is copied. Equivalent to slicing along axis 0.
     * @param index the index along the first dimension
     * @return a new Tensor view with the first dimension removed
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public Tensor slice(long index) {
        return new Tensor(array.slice(index), arena);
    }

    public float getFloat(long... indices) {
        return array.getFloat(indices);
    }

    public void setFloat(float v, long... indices) {
        array.setFloat(v, indices);
    }

    // Package-private NDArray access (for ops layer)
    public NDArray ndarray() {
        return array;
    }

    @Override
    public String toString() {
        return array.toString();
    }

    /**
     * Closes the underlying arena, freeing all off-heap memory for this tensor.
     */
    @Override
    public void close() {
        arena.close();
    }

    // TODO: put them in a better spot (lazy so just adding them in the end - low priority)

    /**
     * Allocating scale operation. Returns a new Tensor
     * where each element is this tensor's element multiplied by {@code v}.
     */
    public Tensor scale(float v) {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.scale(this.array, v, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Allocating ReLU operation. Returns a new Tensor.
     */
    public Tensor relu() {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.relu(this.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Reduction sum operation. Returns a primitive float.
     */
    public float sum() {
        return VectorOps.sum(this.array);
    }

    /**
     * In-Place ReLU operation. Mutates this tensor directly.
     * Notice the method name ends with an underscore (idiomatic for in-place).
     */
    public void relu_() {
        // No new Arena! We modify our own memory directly.
        VectorOps.reluInPlace(this.array);
    }

    /**
     * In-place scale operation. Multiplies every element by {@code v}.
     * Mutates this tensor directly — no allocation.
     */
    public void scale_(float v) {
        VectorOps.scaleInPlace(this.array, v);
    }

    /**
     * Element-wise (Hadamard) product. Returns a new Tensor
     * where each element is the product of the corresponding elements.
     * Both tensors must have the same shape.
     */
    public Tensor mul(Tensor other) {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.mul(this.array, other.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Matrix multiplication. Returns a new Tensor.
     * <p>
     * Supports batched matmul with broadcasting for any dimensionality:
     * <ul>
     *   <li>2D:       (M,K) @ (K,N) → (M,N)
     *   <li>Batched:  (...,M,K) @ (...,K,N) → (...,M,N)
     *   <li>1D @ 1D:  (K,)   @ (K,)   → (1,)       — dot product
     *   <li>1D @ ND:  (K,)   @ (...,K,N) → (...,N)
     *   <li>ND @ 1D:  (...,M,K) @ (K,)   → (...,M)
     * </ul>
     */
    public Tensor matmul(Tensor other) {
        int aRank = this.ndims();
        int bRank = other.ndims();

        // Compute output shape
        long[] outShape;
        if (aRank == 1 && bRank == 1) {
            outShape = new long[]{1};
        } else if (aRank == 1) {
            // (K,) @ (...,K,N) → (...,N)
            long K = this.shape()[0];
            long[] bShape = other.shape();
            if (K != bShape[bRank - 2])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + K + " vs " + bShape[bRank - 2]);
            outShape = new long[bRank - 1];
            System.arraycopy(bShape, 0, outShape, 0, bRank - 2);
            outShape[bRank - 2] = bShape[bRank - 1];
        } else if (bRank == 1) {
            // (...,M,K) @ (K,) → (...,M)
            long K = other.shape()[0];
            long[] aShape = this.shape();
            if (K != aShape[aRank - 1])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + aShape[aRank - 1] + " vs " + K);
            outShape = new long[aRank - 1];
            System.arraycopy(aShape, 0, outShape, 0, aRank - 2);
            outShape[aRank - 2] = aShape[aRank - 2];
        } else {
            // Both >= 2D
            long K = this.shape()[aRank - 1];
            long M = this.shape()[aRank - 2];
            long N = other.shape()[bRank - 1];
            if (K != other.shape()[bRank - 2])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + K + " != " + other.shape()[bRank - 2]);

            int aBatchRank = aRank - 2;
            int bBatchRank = bRank - 2;
            int batchRank = Math.max(aBatchRank, bBatchRank);

            outShape = new long[batchRank + 2];
            for (int i = 0; i < batchRank; i++) {
                long da = i < aBatchRank
                    ? this.shape()[aBatchRank - 1 - i] : 1;
                long db = i < bBatchRank
                    ? other.shape()[bBatchRank - 1 - i] : 1;
                if (da != 1 && db != 1 && da != db)
                    throw new IllegalArgumentException(
                        "Incompatible batch dimensions at axis "
                        + (batchRank - 1 - i) + ": " + da + " vs " + db);
                outShape[batchRank - 1 - i] = Math.max(da, db);
            }
            outShape[batchRank]     = M;
            outShape[batchRank + 1] = N;
        }

        Arena outArena = Arena.ofConfined();
        NDArray out = NDArray.zeros(DType.FLOAT32, outArena, outShape);
        VectorMulOps.matmul(this.array, other.ndarray(), out);
        return new Tensor(out, outArena);
    }

    /**
     * Computes the dot product following NumPy {@code dot} semantics.
     * <p>
     * Handles all dimensionality cases:
     * <ul>
     *   <li>1D @ 1D: scalar-like dot product, returns a {@code [1]} tensor
     *   <li>2D @ 2D: standard matrix multiplication
     *   <li>1D @ ND or ND @ 1D: vector-matrix or matrix-vector multiplication
     *   <li>ND @ ND (both >= 3D): contract the last axis of {@code this} with the
     *       second-to-last axis of {@code other}. The output shape is
     *       {@code this.shape()[:-1] ++ other.shape()[:-2] ++ other.shape()[-1:]},
     *       for example {@code dot((2,3,4,5), (6,5,7))} -> {@code (2,3,4,6,7)}.
     * </ul>
     * <p>
     * Cases where either operand is 1D or 2D are delegated to {@link #matmul(Tensor)}.
     * For the general ND @ ND case, the implementation:
     * <ol>
     *   <li>Reshapes {@code this} to 2D {@code (M, K)} — valid because it is contiguous
     *       with the contracted dimension last
     *   <li>Transposes {@code other} to move the contracted axis to the front, creating
     *       a zero-copy view, then copies it into a contiguous 2D block {@code (K, N)}
     *   <li>Performs a standard 2D matmul via {@link VectorMulOps#matmul}
     *   <li>Reshapes the result to the correct output shape
     * </ol>
     *
     * @param other the tensor to compute the dot product with
     * @return a new tensor containing the dot product
     * @throws IllegalArgumentException if the inner dimensions do not match,
     *                                  if dtypes are not {@link DType#FLOAT32},
     *                                  or if the shapes are otherwise incompatible
     */
    public Tensor dot(Tensor other) {
        int aRank = this.ndims();
        int bRank = other.ndims();

        // Scalar (0-D) tensors are not supported — throw a clear error
        if (aRank == 0 || bRank == 0) {
            throw new IllegalArgumentException(
                "dot product with scalar (0-D) tensors is not supported. "
                + "Use element-wise multiply instead. "
                + "Shapes: this=" + java.util.Arrays.toString(this.shape())
                + ", other=" + java.util.Arrays.toString(other.shape()));
        }

        // Cases 1-3: delegate to matmul (handles 1D@1D, 2D@2D, 1D@ND, ND@1D)
        if (aRank <= 2 || bRank <= 2) {
            return this.matmul(other);
        }

        // Case 4: ND @ ND — general contraction
        long[] aShape = this.shape();
        long[] bShape = other.shape();

        long K = aShape[aRank - 1];
        if (K != bShape[bRank - 2]) {
            throw new IllegalArgumentException(
                "Inner dimensions mismatch for dot product: this.shape[-1] = "
                + K + " != other.shape[-2] = " + bShape[bRank - 2]
                + ". Shapes: this=" + java.util.Arrays.toString(aShape)
                + ", other=" + java.util.Arrays.toString(bShape));
        }

        // Output shape: this.shape[:-1] ++ other.shape[:-2] ++ other.shape[-1:]
        int outRank = aRank - 1 + bRank - 1;  // aRank-1 + bRank-1 (excludes both K dims)
        long[] outShape = new long[outRank];
        System.arraycopy(aShape, 0, outShape, 0, aRank - 1);
        System.arraycopy(bShape, 0, outShape, aRank - 1, bRank - 2);
        outShape[outRank - 1] = bShape[bRank - 1];

        // Flatten A to (M, K) — A is contiguous with contracted dim last, reshape is valid
        long M = 1;
        for (int i = 0; i < aRank - 1; i++) {
            M *= aShape[i];
        }
        NDArray a2d = this.array.reshape(M, K);

        // Transpose B: move contracted axis (bRank-2) to front
        // Permutation: [bRank-2, 0, 1, ..., bRank-3, bRank-1]
        int[] perm = new int[bRank];
        perm[0] = bRank - 2;
        perm[bRank - 1] = bRank - 1;
        for (int i = 1; i < bRank - 1; i++) {
            perm[i] = i - 1;
        }
        NDArray bTransposed = other.ndarray().transpose(perm);

        // Flatten all B dims except the contracted one into N
        // N = product of B.shape[0..bRank-3] * B.shape[bRank-1]
        long N = bShape[bRank - 1];
        for (int i = 0; i < bRank - 2; i++) {
            N *= bShape[i];
        }

        // Copy transposed (non-contiguous) B into a contiguous (K, N) buffer
        Arena copyArena = Arena.ofConfined();
        try {
            NDArray b2d = NDArray.zeros(DType.FLOAT32, copyArena, K, N);

            // Element-wise copy from the transposed view to the contiguous buffer.
            // bTransposed shape after transpose: [K, e0, e1, ..., e_{b-3}, N]
            long[] bIdx = new long[bRank];
            for (long k = 0; k < K; k++) {
                bIdx[0] = k;
                for (long n = 0; n < N; n++) {
                    long tmp = n;
                    for (int d = bRank - 1; d >= 1; d--) {
                        bIdx[d] = tmp % bTransposed.shape[d];
                        tmp /= bTransposed.shape[d];
                    }
                    float val = bTransposed.getFloat(bIdx);
                    b2d.setFloatAtIndex(val, k * N + n);
                }
            }

            // Run standard 2D matmul
            Arena outArena = Arena.ofConfined();
            NDArray out2d = NDArray.zeros(DType.FLOAT32, outArena, M, N);
            VectorMulOps.matmul(a2d, b2d, out2d);

            // Reshape to the correct output shape
            NDArray result = out2d.reshape(outShape);
            return new Tensor(result, outArena);
        } finally {
            copyArena.close();
        }
    }

    public float mean() {
        return VectorOps.mean(this.array);
    }
}