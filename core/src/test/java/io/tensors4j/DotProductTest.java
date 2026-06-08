package io.tensors4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive JUnit 5 tests for:
 *   - NDArray.transpose(int... axes)  — N-dim axis permutation, zero-copy, validation
 *   - NDArray.reshape(long...)       — shape change, contiguous requirement, validation
 *   - Tensor.dot(Tensor)             — NumPy-style dot product (2D, ND, numeric, errors)
 */
class DotProductTest {

    // helper

    private static void fillSequential(NDArray arr) {
        long n = 1;
        for (long d : arr.shape) n *= d;
        for (long i = 0; i < n; i++) arr.setFloatAtIndex((float) i, i);
    }

    private static void fillAll(NDArray arr, float v) {
        long n = 1;
        for (long d : arr.shape) n *= d;
        for (long i = 0; i < n; i++) arr.setFloatAtIndex(v, i);
    }

    // NDArray.transpose(int... axes)

    @Test
    void testTransposePermute3D() {
        // transpose(2,0,1) on shape [2,3,4] → shape [4,2,3]
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            fillSequential(a);

            NDArray t = a.transpose(2, 0, 1);

            assertArrayEquals(new long[]{4, 2, 3}, t.shape,
                    "transpose(2,0,1) of [2,3,4] should produce [4,2,3]");

            // Verify values: T[i][j][k] = source[j][k][i]
            // source[0][0][0] = 0 → T[0][0][0] = 0
            assertEquals(0f, t.getFloat(0, 0, 0), 1e-6f);
            // source[0][0][1] = 1 → T[1][0][0] = 1
            assertEquals(1f, t.getFloat(1, 0, 0), 1e-6f);
            // source[0][1][0] = 4 → T[0][0][1] = 4
            assertEquals(4f, t.getFloat(0, 0, 1), 1e-6f);
            // source[1][0][1] = 12+1 = 13 → T[1][1][0] = 13
            assertEquals(13f, t.getFloat(1, 1, 0), 1e-6f);
            // source[1][2][3] = 12+8+3 = 23 → T[3][1][2] = 23
            assertEquals(23f, t.getFloat(3, 1, 2), 1e-6f);
        }
    }

    @Test
    void testTransposeIsZeroCopy() {
        // Modifying the source must affect the transposed view
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            a.setFloat(10f, 0, 0);
            a.setFloat(20f, 0, 1);
            a.setFloat(30f, 0, 2);
            a.setFloat(40f, 1, 0);
            a.setFloat(50f, 1, 1);
            a.setFloat(60f, 1, 2);

            NDArray t = a.transpose(1, 0);  // shape [3,2]

            // Transposed view should see initial values: T[i][j] = A[j][i]
            assertEquals(10f, t.getFloat(0, 0), 1e-6f);  // A[0][0]
            assertEquals(20f, t.getFloat(1, 0), 1e-6f);  // A[0][1]
            assertEquals(40f, t.getFloat(0, 1), 1e-6f);  // A[1][0]

            // Mutate source
            a.setFloat(99f, 0, 2);
            a.setFloat(77f, 1, 1);

            // Transposed view must reflect the changes
            assertEquals(99f, t.getFloat(2, 0), 1e-6f,
                    "Transposed view did not reflect source mutation — not zero-copy");
            assertEquals(77f, t.getFloat(1, 1), 1e-6f,
                    "Transposed view did not reflect source mutation — not zero-copy");
        }
    }

    @Test
    void testTranspose2DDelegation() {
        // The no arg transpose() should delegate to transpose(1,0)
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 5);
            fillSequential(a);

            NDArray t0 = a.transpose();
            NDArray t1 = a.transpose(1, 0);

            assertArrayEquals(t1.shape, t0.shape);
            assertEquals(t0.getFloat(0, 0), t1.getFloat(0, 0), 1e-6f);
            assertEquals(t0.getFloat(4, 2), t1.getFloat(4, 2), 1e-6f);
        }
    }

    // Validation: wrong length

    @Test
    void testTransposeWrongNumberOfAxesThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            assertThrows(IllegalArgumentException.class, () -> a.transpose(0, 1),
                    "transpose with 2 axes on a 3D array should throw");
            assertThrows(IllegalArgumentException.class, () -> a.transpose(0, 1, 2, 3),
                    "transpose with 4 axes on a 3D array should throw");
        }
    }

    // Validation: out of range

    @Test
    void testTransposeAxisOutOfRangeThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4, 5);
            assertThrows(IllegalArgumentException.class, () -> a.transpose(0, 1, 3),
                    "Axis 3 is out of range [0,2]");
            assertThrows(IllegalArgumentException.class, () -> a.transpose(-1, 0, 1),
                    "Negative axis should throw");
        }
    }

    // Validation: duplicates

    @Test
    void testTransposeDuplicateAxesThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4, 5);
            assertThrows(IllegalArgumentException.class, () -> a.transpose(0, 0, 1),
                    "Duplicate axis 0 should throw");
            assertThrows(IllegalArgumentException.class, () -> a.transpose(1, 2, 1),
                    "Duplicate axis 1 should throw");
        }
    }

    //  NDArray.reshape(long...)

    @Test
    void testReshapeContiguous2Dto1D() {
        // reshape([6]) on [2,3] → shape [6]
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            fillSequential(a);

            NDArray r = a.reshape(6);

            assertArrayEquals(new long[]{6}, r.shape);
            for (long i = 0; i < 6; i++) {
                assertEquals((float) i, r.getFloatAtIndex(i), 1e-6f,
                        "Mismatch at flat index " + i);
            }
        }
    }

    @Test
    void testReshapeContiguous1DtoMultiDim() {
        // reshape([2,2,3]) on [12] → correct values
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 12);
            fillSequential(a);

            NDArray r = a.reshape(2, 2, 3);

            assertArrayEquals(new long[]{2, 2, 3}, r.shape);
            // r[0][0][0] = 0, r[0][0][1] = 1, r[0][0][2] = 2
            // r[0][1][0] = 3, r[0][1][1] = 4, r[0][1][2] = 5
            // r[1][0][0] = 6, ...
            for (long i = 0; i < 2; i++) {
                for (long j = 0; j < 2; j++) {
                    for (long k = 0; k < 3; k++) {
                        float expected = i * 6 + j * 3 + k;
                        assertEquals(expected, r.getFloat(i, j, k), 1e-6f,
                                "Mismatch at [" + i + "," + j + "," + k + "]");
                    }
                }
            }
        }
    }

    @Test
    void testReshapePreservesValues() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            fillSequential(a);
            NDArray r = a.reshape(4, 3);

            // Flat order is preserved — check via flat indices
            for (long i = 0; i < 12; i++) {
                assertEquals(a.getFloatAtIndex(i), r.getFloatAtIndex(i), 1e-6f,
                        "Mismatch at flat index " + i);
            }

            // Spot check multi-index: r[i,j] = a.flat[i*3 + j]
            assertEquals(0f,  r.getFloat(0, 0), 1e-6f); // flat 0
            assertEquals(3f,  r.getFloat(1, 0), 1e-6f); // flat 3
            assertEquals(7f,  r.getFloat(2, 1), 1e-6f); // flat 7
            assertEquals(11f, r.getFloat(3, 2), 1e-6f); // flat 11
        }
    }

    // Validation: non-contiguous reshape throws
    @Test
    void testReshapeNonContiguousThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3, 4);
            NDArray t = a.transpose(1, 0, 2);  // non-contiguous view
            // isContiguous() should return false after transpose
            assertThrows(UnsupportedOperationException.class, () -> t.reshape(24),
                    "Reshaping a non-contiguous (transposed) array should throw");
        }
    }

    @Test
    void testReshapeSlicedNonContiguousThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray s = a.transpose();  // (4,3), non-contiguous
            assertThrows(UnsupportedOperationException.class, () -> s.reshape(12),
                    "Reshaping a transposed (non-contiguous) array should throw");
        }
    }

    // element count mismatch throws

    @Test
    void testReshapeElementCountMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> a.reshape(5),
                    "reshape with 5 elements on a 6-element array should throw");
            assertThrows(IllegalArgumentException.class, () -> a.reshape(2, 4),
                    "reshape to 8 elements on a 6-element array should throw");
            assertThrows(IllegalArgumentException.class, () -> a.reshape(7),
                    "reshape to 7 elements on a 6-element array should throw");
        }
    }

    @Test
    void testReshapeZeroLengthThrows() {
        // Reshape to a shape with zero elements should throw because 0 != 6
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> a.reshape(0),
                    "Reshaping to zero-length should throw due to element count mismatch");
        }
    }

    // Tensor.dot — 2D cases (match matmul)

    @Test
    void testDot2DMatchesMatmul() {
        // dot((3,4), (4,5)) → (3,5) — should produce same result as matmul
        try (Arena arena = Arena.ofConfined()) {
            NDArray aArr = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray bArr = NDArray.zeros(DType.FLOAT32, arena, 4, 5);

            fillSequential(aArr);                         // 0..11
            fillAll(bArr, 1f);

            try (Tensor tA = Tensor.fill(DType.FLOAT32, 0f, 3, 4);
                 Tensor tB = Tensor.fill(DType.FLOAT32, 0f, 4, 5)) {
                for (long i = 0; i < 3; i++)
                    for (long j = 0; j < 4; j++)
                        tA.setFloat(aArr.getFloat(i, j), i, j);
                for (long k = 0; k < 4; k++)
                    for (long j = 0; j < 5; j++)
                        tB.setFloat(1f, k, j);

                try (Tensor dotResult = tA.dot(tB);
                     Tensor mmResult = tA.matmul(tB)) {

                    assertArrayEquals(new long[]{3, 5}, dotResult.shape(),
                            "dot((3,4),(4,5)) should produce shape (3,5)");
                    assertArrayEquals(mmResult.shape(), dotResult.shape(),
                            "dot and matmul output shapes should match for 2D inputs");

                    // Verify numeric values: c[i,j] = sum_k A[i,k] * 1
                    // Row 0: 0+1+2+3 = 6
                    // Row 1: 4+5+6+7 = 22
                    // Row 2: 8+9+10+11 = 38
                    float[] rowSums = {6f, 22f, 38f};
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 5; j++) {
                            float msg = dotResult.getFloat(i, j);
                            assertEquals(rowSums[i], msg, 1e-3f,
                                    "dot mismatch at [" + i + "," + j + "]");
                            assertEquals(mmResult.getFloat(i, j), msg, 1e-3f,
                                    "dot and matmul differ at [" + i + "," + j + "]");
                        }
                    }
                }
            }
        }
    }

    // Tensor.dot ND cases (delegated to matmul)

    @Test
    void testDotND2D_3Ddot2D() {
        // dot((2,3,4), (4,5)) → (2,3,5) ND @ 2D delegates to batched matmul
        try (Tensor a = Tensor.fill(DType.FLOAT32, 0f, 2, 3, 4);
             Tensor b = Tensor.fill(DType.FLOAT32, 0f, 4, 5)) {

            // Fill A with 1s, B with 1s
            fillAll(a.ndarray(), 1f);
            fillAll(b.ndarray(), 1f);

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{2, 3, 5}, result.shape(),
                        "dot((2,3,4),(4,5)) should produce (2,3,5)");

                // Each result cell = sum over K=4 of 1*1 = 4
                for (int b0 = 0; b0 < 2; b0++)
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 5; j++)
                            assertEquals(4f, result.getFloat(b0, i, j), 1e-3f,
                                    "Mismatch at [" + b0 + "," + i + "," + j + "]");
            }
        }
    }

    @Test
    void testDot3Ddot2D() {
        // dot((3,4,5), (5,6)) → (3,4,6)  ND @ 2D delegates to matmul
        try (Tensor a = Tensor.fill(DType.FLOAT32, 0f, 3, 4, 5);
             Tensor b = Tensor.fill(DType.FLOAT32, 0f, 5, 6)) {

            // Fill A with sequential values
            long idx = 1;
            for (int b0 = 0; b0 < 3; b0++)
                for (int i = 0; i < 4; i++)
                    for (int k = 0; k < 5; k++)
                        a.setFloat(idx++, b0, i, k);

            // Fill B with all 1s
            for (int k = 0; k < 5; k++)
                for (int j = 0; j < 6; j++)
                    b.setFloat(1f, k, j);

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{3, 4, 6}, result.shape(),
                        "dot((3,4,5),(5,6)) should produce (3,4,6)");

                // Each result cell = sum of A's row over the K dimension
                for (int b0 = 0; b0 < 3; b0++) {
                    for (int i = 0; i < 4; i++) {
                        // Row values in batch b0, row i: 5 values starting at 1 + b0*20 + i*5
                        float expected = 0;
                        for (int k = 0; k < 5; k++)
                            expected += a.getFloat(b0, i, k);
                        for (int j = 0; j < 6; j++)
                            assertEquals(expected, result.getFloat(b0, i, j), 1e-3f,
                                    "Mismatch at [" + b0 + "," + i + "," + j + "]");
                    }
                }
            }
        }
    }

    @Test
    void testDot2DdotND() {
        // dot((2,3), (4,3,5)) → (4,2,5) — 2D @ ND delegates to matmul with broadcasting
        try (Tensor a = Tensor.fill(DType.FLOAT32, 0f, 2, 3);
             Tensor b = Tensor.fill(DType.FLOAT32, 0f, 4, 3, 5)) {

            // Fill A with sequential 1..6
            long idx = 1;
            for (int i = 0; i < 2; i++)
                for (int k = 0; k < 3; k++)
                    a.setFloat(idx++, i, k);

            // Fill B with all 1s
            for (int b0 = 0; b0 < 4; b0++)
                for (int k = 0; k < 3; k++)
                    for (int j = 0; j < 5; j++)
                        b.setFloat(1f, b0, k, j);

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{4, 2, 5}, result.shape(),
                        "dot((2,3),(4,3,5)) should produce (4,2,5)");

                // Each result[b0,i,j] = sum_k A[i,k] * 1 = sum of row i of A
                // A row 0: 1+2+3 = 6
                // A row 1: 4+5+6 = 15
                float[] rowSums = {6f, 15f};
                for (int b0 = 0; b0 < 4; b0++)
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 5; j++)
                            assertEquals(rowSums[i], result.getFloat(b0, i, j), 1e-3f,
                                    "Mismatch at [" + b0 + "," + i + "," + j + "]");
            }
        }
    }

    // ====================================================================
    //  Tensor.dot — ND @ ND (general contraction, both >= 3D)
    // ====================================================================

    @Test
    void testDotNDdotND_4Ddot3D() {
        // dot((2,3,4,5), (6,5,7)) → (2,3,4,6,7)
        // The general ND contraction path: this[-1] must equal other[-2]
        try (Tensor a = Tensor.fill(DType.FLOAT32, 0f, 2, 3, 4, 5);
             Tensor b = Tensor.fill(DType.FLOAT32, 0f, 6, 5, 7)) {

            // Fill A with sequential values 1..120
            long idx = 1;
            for (int b0 = 0; b0 < 2; b0++)
                for (int b1 = 0; b1 < 3; b1++)
                    for (int b2 = 0; b2 < 4; b2++)
                        for (int k = 0; k < 5; k++)
                            a.setFloat(idx++, b0, b1, b2, k);

            // Fill B with all 1s
            for (int d0 = 0; d0 < 6; d0++)
                for (int k = 0; k < 5; k++)
                    for (int d2 = 0; d2 < 7; d2++)
                        b.setFloat(1f, d0, k, d2);

            try (Tensor result = a.dot(b)) {
                long[] expectedShape = {2, 3, 4, 6, 7};
                assertArrayEquals(expectedShape, result.shape(),
                        "dot((2,3,4,5),(6,5,7)) should produce (2,3,4,6,7)");

                // Each result cell = sum over K=5 of A values in that row
                // A is reshaped to (2*3*4, 5) = (24, 5), B to (5, 6*7) = (5, 42)
                // Result cell [b0,b1,b2,d0,d2] = sum_{k=0..4} A[b0,b1,b2,k]
                for (int b0 = 0; b0 < 2; b0++) {
                    for (int b1 = 0; b1 < 3; b1++) {
                        for (int b2 = 0; b2 < 4; b2++) {
                            // Sum of the 5 values in this "row"
                            float rowSum = 0;
                            for (int k = 0; k < 5; k++)
                                rowSum += a.getFloat(b0, b1, b2, k);

                            for (int d0 = 0; d0 < 6; d0++) {
                                for (int d2 = 0; d2 < 7; d2++) {
                                    assertEquals(rowSum, result.getFloat(b0, b1, b2, d0, d2), 1e-3f,
                                            "Mismatch at [" + b0 + "," + b1 + "," + b2 + "," + d0 + "," + d2 + "]");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void testDotNDdotND_AllOnesNumeric() {
        // dot((2,2,3), (4,3,5)) → (2,2,4,5) — all ones, each cell = K = 3
        try (Tensor a = Tensor.ones(DType.FLOAT32, 2, 2, 3);
             Tensor b = Tensor.ones(DType.FLOAT32, 4, 3, 5)) {

            try (Tensor result = a.dot(b)) {
                long[] expectedShape = {2, 2, 4, 5};
                assertArrayEquals(expectedShape, result.shape());

                // All-ones: each result cell = sum over K=3 of 1*1 = 3
                for (int b0 = 0; b0 < 2; b0++)
                    for (int b1 = 0; b1 < 2; b1++)
                        for (int d0 = 0; d0 < 4; d0++)
                            for (int d1 = 0; d1 < 5; d1++)
                                assertEquals(3f, result.getFloat(b0, b1, d0, d1), 1e-3f);
            }
        }
    }

    // ====================================================================
    //  Tensor.dot — error handling
    // ====================================================================

    @Test
    void testDotInnerDimMismatchThrows() {
        // 2D case: (3,4) dot (5,6) → inner dims 4 != 5
        try (Tensor a = Tensor.ones(DType.FLOAT32, 3, 4);
             Tensor b = Tensor.ones(DType.FLOAT32, 5, 6)) {

            assertThrows(IllegalArgumentException.class, () -> a.dot(b),
                    "Inner dimension mismatch (4 != 5) should throw");
        }
    }

    @Test
    void testDotNDInnerDimMismatchThrows() {
        // ND @ ND: (2,3,5) dot (4,6,7) → 5 != 6
        try (Tensor a = Tensor.ones(DType.FLOAT32, 2, 3, 5);
             Tensor b = Tensor.ones(DType.FLOAT32, 4, 6, 7)) {

            assertThrows(IllegalArgumentException.class, () -> a.dot(b),
                    "Inner dimension mismatch in ND case (5 != 6) should throw");
        }
    }

    @Test
    void testDot1Ddot1D() {
        // dot((3,), (3,)) → (1,) — dot product, delegates to matmul
        try (Tensor a = Tensor.fill(DType.FLOAT32, 0f, 3);
             Tensor b = Tensor.fill(DType.FLOAT32, 0f, 3)) {

            a.setFloat(1f, 0);
            a.setFloat(2f, 1);
            a.setFloat(3f, 2);
            b.setFloat(4f, 0);
            b.setFloat(5f, 1);
            b.setFloat(6f, 2);

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{1}, result.shape());
                assertEquals(32f, result.getFloat(0), 1e-4f); // 1*4 + 2*5 + 3*6 = 32
            }
        }
    }

    @Test
    void testDot1DdotND() {
        // dot((4,), (4,5)) → (5,) — vector @ matrix, delegates to matmul
        try (Tensor a = Tensor.fill(DType.FLOAT32, 1f, 4);
             Tensor b = Tensor.ones(DType.FLOAT32, 4, 5)) {

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{5}, result.shape());
                // Each result[j] = sum_k a[k] * b[k,j] = 1*1*4 = 4
                for (int j = 0; j < 5; j++)
                    assertEquals(4f, result.getFloat(j), 1e-3f);
            }
        }
    }

    @Test
    void testDotNDdot1D() {
        // dot((3,4), (4,)) → (3,) — matrix @ vector, delegates to matmul
        try (Tensor a = Tensor.ones(DType.FLOAT32, 3, 4);
             Tensor b = Tensor.fill(DType.FLOAT32, 1f, 4)) {

            try (Tensor result = a.dot(b)) {
                assertArrayEquals(new long[]{3}, result.shape());
                // Each result[i] = sum_k a[i,k] * b[k] = 1*1*4 = 4
                for (int i = 0; i < 3; i++)
                    assertEquals(4f, result.getFloat(i), 1e-3f);
            }
        }
    }
}
