package io.tensors4j;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorOpsTest {

    // Helper to inject specific values into an NDArray for testing
    private void setValues(NDArray arr, float... values) {
        for (int i = 0; i < values.length; i++) {
            arr.segment.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
    }

    // Helper to assert an entire NDArray matches expected values
    private void assertArrayEquals(float[] expected, NDArray actual) {
        for (int i = 0; i < expected.length; i++) {
            float val = actual.segment.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            assertEquals(expected[i], val, 1e-5f, "Mismatch at index " + i);
        }
    }

    @Test
    void testAllocatingBinaryOps() {
        try (Arena arena = Arena.ofConfined()) {
            // Size 11 forces both the SIMD loop (8) and the Scalar Tail (3)
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 11);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 11);

            setValues(a, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f);
            setValues(b, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f);

            // Test Add
            NDArray sum = VectorOps.add(a, b, arena);
            assertArrayEquals(new float[]{3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f}, sum);

            // Test Sub
            NDArray diff = VectorOps.sub(a, b, arena);
            assertArrayEquals(new float[]{-1f, 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, diff);

            // Test Mul
            NDArray prod = VectorOps.mul(a, b, arena);
            assertArrayEquals(new float[]{2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f}, prod);
        }
    }

    @Test
    void testAllocatingUnaryOps() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 11);
            setValues(a, -5f, -4f, -3f, -2f, -1f, 0f, 1f, 2f, 3f, 4f, 5f);

            // Test Scale
            NDArray scaled = VectorOps.scale(a, 2.0f, arena);
            assertArrayEquals(new float[]{-10f, -8f, -6f, -4f, -2f, 0f, 2f, 4f, 6f, 8f, 10f}, scaled);

            // Test ReLU (should flatten negatives to 0)
            NDArray activated = VectorOps.relu(a, arena);
            assertArrayEquals(new float[]{0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f}, activated);
        }
    }

    @Test
    void testInPlaceOps() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 11);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 11);

            setValues(a, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f);
            setValues(b, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f);

            // Test Add In-Place
            VectorOps.addInPlace(a, b);
            assertArrayEquals(new float[]{11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f}, a);

            // Test Scale In-Place
            VectorOps.scaleInPlace(a, -1f);
            assertArrayEquals(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f, -21f}, a);

            // Test ReLU In-Place (all are negative now, should become 0)
            VectorOps.reluInPlace(a);
            assertArrayEquals(new float[]{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, a);
        }
    }

    @Test
    void testReductions() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 11);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 11);

            // 1 through 11
            setValues(a, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f);

            // All 2s
            setValues(b, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f);

            // Sum of 1..11 is (11 * 12) / 2 = 66
            float expectedSum = 66f;
            assertEquals(expectedSum, VectorOps.sum(a), 1e-5f);

            // Mean of 1..11 is 66 / 11 = 6
            float expectedMean = 6f;
            assertEquals(expectedMean, VectorOps.mean(a), 1e-5f);

            // Dot product: sum(a[i] * 2) = 66 * 2 = 132
            float expectedDot = 132f;
            assertEquals(expectedDot, VectorOps.dot(a, b), 1e-5f);
        }
    }

    @Test
    void testShapeMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 10);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 5); // Different shape

            assertThrows(IllegalArgumentException.class, () -> {
                VectorOps.add(a, b, arena);
            }, "Expected shape mismatch to throw IllegalArgumentException");
        }
    }
}
