package io.tensors4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * F32 matmul for any rank ≥ 2 (batched) and 1D dot-product.
 * <p>
 * Follows NumPy/PyTorch broadcasting semantics for leading dimensions.
 * <ul>
 *   <li>Both 2D:       (M,K) @ (K,N) → (M,N)                   — standard matmul
 *   <li>Both ≥ 2D:     (...,M,K) @ (...,K,N) → (...,M,N)       — batched, broadcast
 *   <li>1D @ 1D:       (K,)   @ (K,)   → (1,)                  — dot product
 *   <li>1D @ ND:       (K,)   @ (...,K,N) → (...,N)            — prepend 1
 *   <li>ND @ 1D:       (...,M,K) @ (K,)   → (...,M)            — append 1
 * </ul>
 */
public final class VectorMulOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder BO = ByteOrder.nativeOrder();
    private static final int TILE = 64;

    private VectorMulOps() {}

    //  Public entry point
    /**
     * C = A @ B.  Supports batched/broadcast matmul for any dimensionality.
     * <p>
     * {@code c} must be pre-allocated with the correct output shape.
     * All arrays must be F32, contiguous, and (except size-1 broadcast dims)
     * row-major contiguous.
     */
    public static void matmul(NDArray a, NDArray b, NDArray c) {
        checkF32(a); checkF32(b); checkF32(c);
        checkContiguous(a); checkContiguous(b); checkContiguous(c);

        int aRank = a.shape.length;
        int bRank = b.shape.length;

        // 1D cases
        if (aRank == 1 && bRank == 1) { matmul1D1D(a, b, c); return; }
        if (aRank == 1)               { matmul1DND(a, b, c); return; }
        if (bRank == 1)               { matmulND1D(a, b, c); return; }

        // Both ≥ 2D — validate inner dims
        int aKdim = aRank - 1;
        int bKdim = bRank - 2;
        long K = a.shape[aKdim];
        if (K != b.shape[bKdim])
            throw new IllegalArgumentException(
                "Inner dimension mismatch: A shape " + Arrays.toString(a.shape)
                + " vs B shape " + Arrays.toString(b.shape));

        long M = a.shape[aRank - 2];
        long N = b.shape[bRank - 1];

        // Broadcast batch dimensions
        int aBatchRank = aRank - 2;
        int bBatchRank = bRank - 2;
        int batchRank  = Math.max(aBatchRank, bBatchRank);

        long[] batchShape = new long[batchRank];
        for (int i = 0; i < batchRank; i++) {
            // right align: index from the end
            long da = i < aBatchRank ? a.shape[aBatchRank - 1 - i] : 1;
            long db = i < bBatchRank ? b.shape[bBatchRank - 1 - i] : 1;
            if (da != 1 && db != 1 && da != db)
                throw new IllegalArgumentException(
                    "Incompatible batch dimensions at axis "
                    + (batchRank - 1 - i) + ": " + da + " vs " + db);
            batchShape[batchRank - 1 - i] = Math.max(da, db);
        }

        // verify output shape
        long[] cExpected = new long[batchRank + 2];
        System.arraycopy(batchShape, 0, cExpected, 0, batchRank);
        cExpected[batchRank]     = M;
        cExpected[batchRank + 1] = N;
        if (!Arrays.equals(c.shape, cExpected))
            throw new IllegalArgumentException(
                "Output shape mismatch: expected " + Arrays.toString(cExpected)
                + " but got " + Arrays.toString(c.shape));

        // Per slice tiled matmul
        // Flatten batch iteration
        long batchTotal = NDArray.elementCount(batchShape);

        long[] idx = new long[batchRank];
        for (long flat = 0; flat < batchTotal; flat++) {
            // Unflatten
            long tmp = flat;
            for (int d = batchRank - 1; d >= 0; d--) {
                idx[d] = tmp % batchShape[d];
                tmp /= batchShape[d];
            }

            long aOff = a.offset;
            for (int d = 0; d < aBatchRank; d++) {
                long bi = idx[d + (batchRank - aBatchRank)];
                if (a.shape[d] == 1) bi = 0;
                aOff += bi * a.strides[d];
            }

            long bOff = b.offset;
            for (int d = 0; d < bBatchRank; d++) {
                long bi = idx[d + (batchRank - bBatchRank)];
                if (b.shape[d] == 1) bi = 0;
                bOff += bi * b.strides[d];
            }

            long cOff = c.offset;
            for (int d = 0; d < batchRank; d++) {
                cOff += idx[d] * c.strides[d];
            }

            tiledKernel(a.segment, aOff, b.segment, bOff, c.segment, cOff,
                        (int) M, (int) K, (int) N);
        }
    }

    //  1D helpers
    /** (K,) @ (K,) → (1,) — dot product into a 1-element output. */
    private static void matmul1D1D(NDArray a, NDArray b, NDArray c) {
        if (a.shape[0] != b.shape[0])
            throw new IllegalArgumentException("Dot shape mismatch: "
                + a.shape[0] + " vs " + b.shape[0]);
        if (c.shape.length != 1 || c.shape[0] != 1)
            throw new IllegalArgumentException(
                "1D@1D output must be shape [1]");
        c.segment.set(ValueLayout.JAVA_FLOAT, c.offset, dotKernel(a, b));
    }

    /** (K,) @ (...,K,N) → (...,N).
     *  Implemented as (1,K) @ (K,N) via the tiled kernel, then mapped to 1D output. */
    private static void matmul1DND(NDArray a, NDArray b, NDArray c) {
        int bRank = b.shape.length;
        long K = a.shape[0];
        if (K != b.shape[bRank - 2])
            throw new IllegalArgumentException("Inner dim mismatch");
        long N = b.shape[bRank - 1];

        // c shape should be b's batch dims + [N]
        int cBatchRank = bRank - 2;
        long[] cShape = new long[cBatchRank + 1];
        System.arraycopy(b.shape, 0, cShape, 0, cBatchRank);
        cShape[cBatchRank] = N;
        if (!Arrays.equals(c.shape, cShape))
            throw new IllegalArgumentException("Output shape mismatch");

        long bBatchFlat = 1;
        for (int d = 0; d < cBatchRank; d++) bBatchFlat *= b.shape[d];

        long[] idx = new long[cBatchRank];
        for (long flat = 0; flat < bBatchFlat; flat++) {
            long tmp = flat;
            for (int d = cBatchRank - 1; d >= 0; d--) {
                idx[d] = tmp % cShape[d];
                tmp /= cShape[d];
            }

            long bOff = b.offset;
            for (int d = 0; d < cBatchRank; d++) {
                bOff += idx[d] * b.strides[d];
            }
            long cOff = c.offset;
            for (int d = 0; d < cBatchRank; d++) {
                cOff += idx[d] * c.strides[d];
            }

            // Treat a as (1,K) and call the tiled kernel with M=1
            // The kernel does: for i=0 (just once), for each k:
            //   aVal = a[k] at a.offset + k*4
            //   for j: c[j] += aVal * b[k][j]
            tiledKernel(a.segment, a.offset, b.segment, bOff, c.segment, cOff,
                        1, (int) K, (int) N);
        }
    }

    /** (...,M,K) @ (K,) → (...,M).
     *  Implemented as (M,K) @ (K,1) via the tiled kernel, then mapped to 1D output. */
    private static void matmulND1D(NDArray a, NDArray b, NDArray c) {
        int aRank = a.shape.length;
        long K = a.shape[aRank - 1];
        if (K != b.shape[0])
            throw new IllegalArgumentException("Inner dim mismatch");
        long M = a.shape[aRank - 2];

        int cBatchRank = aRank - 2;
        long[] cShape = new long[cBatchRank + 1];
        System.arraycopy(a.shape, 0, cShape, 0, cBatchRank);
        cShape[cBatchRank] = M;
        if (!Arrays.equals(c.shape, cShape))
            throw new IllegalArgumentException("Output shape mismatch");

        long aBatchFlat = 1;
        for (int d = 0; d < cBatchRank; d++) aBatchFlat *= a.shape[d];

        long[] idx = new long[cBatchRank];
        for (long flat = 0; flat < aBatchFlat; flat++) {
            long tmp = flat;
            for (int d = cBatchRank - 1; d >= 0; d--) {
                idx[d] = tmp % cShape[d];
                tmp /= cShape[d];
            }

            long aOff = a.offset;
            for (int d = 0; d < cBatchRank; d++) {
                aOff += idx[d] * a.strides[d];
            }
            long cOff = c.offset;
            for (int d = 0; d < cBatchRank; d++) {
                cOff += idx[d] * c.strides[d];
            }

            // Treat b as (K,1) and call the tiled kernel with N=1
            //   aVal = a[i][k] at aOff + (i*K + k)*4
            //   for j=0 (just once): c[i] += aVal * b[k]
            tiledKernel(a.segment, aOff, b.segment, b.offset, c.segment, cOff,
                        (int) M, (int) K, 1);
        }
    }

    //  Dot-product helper (used by 1D@1D)
    /**
     * Flat dot product of two contiguous F32 arrays.  Same as VectorOps.dot.
     */
    private static float dotKernel(NDArray a, NDArray b) {
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();

        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;
        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO);
            acc = va.fma(vb, acc);
            i += step;
        }

        float total = acc.reduceLanes(VectorOperators.ADD);

        if (i < n) {
            VectorMask<Float> mask = SPECIES.indexInRange(i, n);
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO, mask);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO, mask);
            total += va.mul(vb).reduceLanes(VectorOperators.ADD, mask);
        }
        return total;
    }

    //  Tiled 2D kernel  (core building block)
    /**
     * 3-level tiled matmul kernel for one 2D slice.
     * Operates on flat byte offsets — caller supplies slice base offsets.
     */
    private static void tiledKernel(
            MemorySegment aSeg, long aBaseOff,
            MemorySegment bSeg, long bBaseOff,
            MemorySegment cSeg, long cBaseOff,
            int M, int K, int N) {

        int step = SPECIES.length();

        for (int i0 = 0; i0 < M; i0 += TILE) {
            int iEnd = Math.min(i0 + TILE, M);
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int kEnd = Math.min(k0 + TILE, K);
                for (int j0 = 0; j0 < N; j0 += TILE) {
                    int jEnd = Math.min(j0 + TILE, N);

                    for (int i = i0; i < iEnd; i++) {
                        for (int k = k0; k < kEnd; k++) {
                            float aVal = aSeg.get(ValueLayout.JAVA_FLOAT,
                                    aBaseOff + ((long) i * K + k) * Float.BYTES);
                            FloatVector va = FloatVector.broadcast(SPECIES, aVal);

                            int j = j0;
                            int jBound = j0 + SPECIES.loopBound(jEnd - j0);
                            for (; j < jBound; j += step) {
                                long bOff = bBaseOff + ((long) k * N + j) * Float.BYTES;
                                long cOff = cBaseOff + ((long) i * N + j) * Float.BYTES;
                                FloatVector vb = FloatVector.fromMemorySegment(SPECIES, bSeg, bOff, BO);
                                FloatVector vc = FloatVector.fromMemorySegment(SPECIES, cSeg, cOff, BO);
                                va.fma(vb, vc).intoMemorySegment(cSeg, cOff, BO);
                            }
                            for (; j < jEnd; j++) {
                                long bOff = bBaseOff + ((long) k * N + j) * Float.BYTES;
                                long cOff = cBaseOff + ((long) i * N + j) * Float.BYTES;
                                float prev = cSeg.get(ValueLayout.JAVA_FLOAT, cOff);
                                cSeg.set(ValueLayout.JAVA_FLOAT, cOff,
                                        prev + aVal * bSeg.get(ValueLayout.JAVA_FLOAT, bOff));
                            }
                        }
                    }
                }
            }
        }
    }

    // helpers
    private static void checkF32(NDArray a) {
        if (a.dtype != DType.FLOAT32)
            throw new IllegalArgumentException(
                "matmul requires FLOAT32, got " + a.dtype);
    }

    private static void checkContiguous(NDArray a) {
        if (!a.isContiguous())
            throw new UnsupportedOperationException(
                "Non-contiguous arrays not supported. Call .compact() first.");
    }
}
