package io.tensors4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.*;

class VectorMulOpsTest {

    // Helpers

    private void fill(NDArray arr, float... vals) {
        for (int i = 0; i < vals.length; i++)
            arr.setFloatAtIndex(vals[i], i);
    }

    @Test
    void testSmallMatmulCorrectness() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 3, 2);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 2);

            fill(a, 1f, 2f, 3f, 4f, 5f, 6f);
            fill(b, 7f, 8f, 9f, 10f, 11f, 12f);

            VectorMulOps.matmul(a, b, c);

            assertEquals(58f,  c.getFloat(0, 0), 1e-4f);
            assertEquals(64f,  c.getFloat(0, 1), 1e-4f);
            assertEquals(139f, c.getFloat(1, 0), 1e-4f);
            assertEquals(154f, c.getFloat(1, 1), 1e-4f);
        }
    }

    @Test
    void testIdentityMatmul() {
        try (Arena arena = Arena.ofConfined()) {
            int N = 4;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray identity = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, N, N);

            for (int i = 0; i < N * N; i++) a.setFloatAtIndex(i + 1f, i);
            for (int i = 0; i < N; i++) identity.setFloat(1f, i, i);

            VectorMulOps.matmul(a, identity, c);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    assertEquals(a.getFloat(i, j), c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testNonMultipleOfSIMDWidth() {
        try (Arena arena = Arena.ofConfined()) {
            int M = 3, K = 5, N = 3;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, M, K);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, K, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, M, N);

            for (int i = 0; i < M * K; i++) a.setFloatAtIndex(1f, i);
            for (int i = 0; i < K * N; i++) b.setFloatAtIndex(1f, i);

            VectorMulOps.matmul(a, b, c);

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    assertEquals((float) K, c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testLargerThanTileSize() {
        try (Arena arena = Arena.ofConfined()) {
            int N = 100;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, N, N);

            for (int i = 0; i < N * N; i++) a.setFloatAtIndex(1f, i);
            for (int i = 0; i < N; i++) b.setFloat(1f, i, i);

            VectorMulOps.matmul(a, b, c);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    assertEquals(1f, c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testShapeMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 5, 3);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3, 3);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }

    // ======================================================================
    //  Batched (3D) matmul tests
    // ======================================================================

    @Test
    void testBatched3DSameBatchSize() {
        // (2, 3, 4) @ (2, 4, 5) → (2, 3, 5)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 4, 5);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 5);

            // Fill a: batch0 = [[1..12]], batch1 = [[13..24]]
            for (int b0 = 0; b0 < 2; b0++)
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 4; k++)
                        a.setFloat(b0 * 12 + i * 4 + k + 1f, b0, i, k);

            // Fill b: all 1s → each result cell = sum of a's row = 1+2+3+4 = 10
            for (int b0 = 0; b0 < 2; b0++)
                for (int k = 0; k < 4; k++)
                    for (int j = 0; j < 5; j++)
                        b.setFloat(1f, b0, k, j);

            VectorMulOps.matmul(a, b, c);

            for (int b0 = 0; b0 < 2; b0++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 5; j++) {
                        // Each row sum in batch b0 row i: 4 values starting at offset
                        float row0 = b0 * 12 + i * 4 + 1;
                        float expected = row0 + (row0 + 1) + (row0 + 2) + (row0 + 3);
                        assertEquals(expected, c.getFloat(b0, i, j), 1e-3f,
                                "Mismatch at [" + b0 + "," + i + "," + j + "]");
                    }
                }
            }
        }
    }

    @Test
    void testBatched3DNonMultipleSIMD() {
        // (2, 3, 5) @ (2, 5, 3) = (2, 3, 3) -> 5 is not a SIMD multiple
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 5);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 5, 3);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 3);

            for (int b0 = 0; b0 < 2; b0++)
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 5; k++)
                        a.setFloat(b0 * 15 + i * 5 + k + 1f, b0, i, k);

            for (int b0 = 0; b0 < 2; b0++)
                for (int k = 0; k < 5; k++)
                    for (int j = 0; j < 3; j++)
                        b.setFloat(1f, b0, k, j);

            VectorMulOps.matmul(a, b, c);

            for (int b0 = 0; b0 < 2; b0++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        float row0 = b0 * 15 + i * 5 + 1;
                        float expected = row0 + (row0+1) + (row0+2) + (row0+3) + (row0+4);
                        assertEquals(expected, c.getFloat(b0, i, j), 1e-3f,
                                "Mismatch at [" + b0 + "," + i + "," + j + "]");
                    }
                }
            }
        }
    }

    // ======================================================================
    //  4D batched matmul tests
    // ======================================================================

    @Test
    void testBatched4D() {
        // (2, 3, 4, 5) @ (2, 3, 5, 6) → (2, 3, 4, 6)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4, 5);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 5, 6);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4, 6);

            // Fill a with sequential values
            long val = 1;
            for (int b0 = 0; b0 < 2; b0++)
                for (int b1 = 0; b1 < 3; b1++)
                    for (int i = 0; i < 4; i++)
                        for (int k = 0; k < 5; k++)
                            a.setFloat(val++, b0, b1, i, k);

            // B all ones -> each result cell = sum of As row over K
            for (int b0 = 0; b0 < 2; b0++)
                for (int b1 = 0; b1 < 3; b1++)
                    for (int k = 0; k < 5; k++)
                        for (int j = 0; j < 6; j++)
                            b.setFloat(1f, b0, b1, k, j);

            VectorMulOps.matmul(a, b, c);

            // Verify each (b0,b1,i,j) = sum of a's row at that batch+row
            for (int b0 = 0; b0 < 2; b0++) {
                for (int b1 = 0; b1 < 3; b1++) {
                    for (int i = 0; i < 4; i++) {
                        float rowSum = 0;
                        for (int k = 0; k < 5; k++)
                            rowSum += a.getFloat(b0, b1, i, k);
                        for (int j = 0; j < 6; j++) {
                            assertEquals(rowSum, c.getFloat(b0, b1, i, j), 1e-3f,
                                    "Mismatch at [" + b0 + "," + b1 + "," + i + "," + j + "]");
                        }
                    }
                }
            }
        }
    }

    // ======================================================================
    //  Broadcasting tests
    // ======================================================================

    @Test
    void testBroadcastBatchDim2Dvs3D() {
        // (3, 4) @ (2, 4, 5) → (2, 3, 5) A is 2D, broadcast over batch
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 4, 5);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 5);

            // A: [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 4; k++)
                    a.setFloat(i + 1f, i, k);

            // B: all 1s
            for (int b0 = 0; b0 < 2; b0++)
                for (int k = 0; k < 4; k++)
                    for (int j = 0; j < 5; j++)
                        b.setFloat(1f, b0, k, j);

            VectorMulOps.matmul(a, b, c);

            // Each result cell = sum of As row (which is 4 * (i+1))
            for (int b0 = 0; b0 < 2; b0++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 5; j++)
                        assertEquals(4f * (i + 1), c.getFloat(b0, i, j), 1e-3f);
        }
    }

    @Test
    void testBroadcastWithSize1Dim() {
        // (2, 1, 3, 4) @ (1, 4, 5) → (2, 1, 3, 5) broadcasting with size-1 dims
        // Actually the result should be (2, 1, 3, 5) not merging further since batchRank = 2 vs 1
        // Wait: a batch dims = [2,1] (rank 4-2=2), b batch dims = [1] (rank 3-2=1)
        // max=2, right-align: dim0: max(2,1)=2, dim1: max(1,1)=1 → batch=[2,1]
        // So output = (2, 1, 3, 5)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 1, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 1, 4, 5);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 1, 3, 5);

            // A: simple fill
            for (int b0 = 0; b0 < 2; b0++)
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 4; k++)
                        a.setFloat(b0 * 12 + i * 4 + k + 1f, b0, 0, i, k);

            // B: all 1s
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 5; j++)
                    b.setFloat(1f, 0, k, j);

            VectorMulOps.matmul(a, b, c);

            // Each row = sum of row values (same logic as 2D case)
            for (int b0 = 0; b0 < 2; b0++) {
                for (int i = 0; i < 3; i++) {
                    float rowSum = 0;
                    for (int k = 0; k < 4; k++)
                        rowSum += a.getFloat(b0, 0, i, k);
                    for (int j = 0; j < 5; j++) {
                        assertEquals(rowSum, c.getFloat(b0, 0, i, j), 1e-3f,
                                "Mismatch at [" + b0 + ",0," + i + "," + j + "]");
                    }
                }
            }
        }
    }

    // ======================================================================
    //  1D matmul tests
    // ======================================================================

    @Test
    void test1Ddot1D() {
        // (3,) @ (3,) = (1,)  — dot product
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 3);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 1);

            fill(a, 1f, 2f, 3f);
            fill(b, 4f, 5f, 6f);

            VectorMulOps.matmul(a, b, c);

            assertEquals(4 + 2*5 + 3*6, c.getFloat(0), 1e-4f);
        }
    }

    @Test
    void test1Ddot1DThrowsOnShapeMismatch() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 4);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 1);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }

    @Test
    void test1Dmatmul2D() {
        // (4,) @ (4,5) → (5,)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 4, 5);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 5);

            fill(a, 1f, 2f, 3f, 4f);
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 5; j++)
                    b.setFloat((k + 1f) * (j + 1f), k, j);

            VectorMulOps.matmul(a, b, c);

            // c[j] = sum_k a[k] * b[k,j]
            //      = 1*(j+1) + 2*2*(j+1) + 3*3*(j+1) + 4*4*(j+1)
            //      = (j+1) * (1 + 4 + 9 + 16) = (j+1) * 30
            for (int j = 0; j < 5; j++) {
                float expected = (j + 1f) * (1 + 4 + 9 + 16);
                assertEquals(expected, c.getFloat(j), 1e-3f,
                        "Mismatch at [" + j + "]");
            }
        }
    }

    @Test
    void test2Dmatmul1D() {
        // (3,4) @ (4,) → (3,)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 4);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3);

            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 4; k++)
                    a.setFloat((i + 1f) * (k + 1f), i, k);
            fill(b, 1f, 2f, 3f, 4f);

            VectorMulOps.matmul(a, b, c);

            // c[i] = sum_k a[i,k] * b[k]
            //      = (i+1) * (1*1 + 2*2 + 3*3 + 4*4) = (i+1) * 30
            for (int i = 0; i < 3; i++)
                assertEquals((i + 1f) * 30f, c.getFloat(i), 1e-3f,
                        "Mismatch at [" + i + "]");
        }
    }

    // ======================================================================
    //  Error cases for higher dims
    // ======================================================================

    @Test
    void testBatchedInnerDimMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 5, 6); // K=4 vs K=5
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 6);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }

    @Test
    void testBatchedIncompatibleBatchDimsThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4, 5);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 5, 6); // batch 3 vs 2
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3, 4, 6);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }

    @Test
    void testOutputShapeMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 2, 4, 5);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3, 3, 5); // wrong batch size

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }
}