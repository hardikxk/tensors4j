package io.tensors4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorTest {
    @Test void zeros() {
        try (var t = Tensor.zeros(DType.FLOAT32, 3, 4)) {
            assertEquals(12, t.size());
            assertEquals(0f, t.getFloat(2, 3));
        }
    }
    @Test void ones() {
        try (var t = Tensor.ones(DType.FLOAT32, 2, 3)) {
            assertEquals(1f, t.getFloat(1, 2));
        }
    }
    @Test void fill() {
        try (var t = Tensor.fill(DType.FLOAT32, 7f, 4, 4)) {
            assertEquals(7f, t.getFloat(3, 3));
        }
    }
    @Test void setRoundTrip() {
        try (var t = Tensor.zeros(DType.FLOAT32, 3, 3)) {
            t.setFloat(42f, 1, 1);
            assertEquals(42f, t.getFloat(1, 1));
            assertEquals(0f,  t.getFloat(0, 0));
        }
    }
    @Test void transposeIsView() {
        try (var t = Tensor.fill(DType.FLOAT32, 0f, 2, 3)) {
            t.setFloat(9f, 0, 2);
            var tr = t.ndarray().transpose();
            assertEquals(9f, tr.getFloat(2, 0)); // zero-copy view
        }
    }
}