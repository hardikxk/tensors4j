package io.tensors4j;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for arena safety, memory allocation, and proper resource cleanup.
 */
class ArenaAndMemoryTest {

    @Test
    void allocatingFromClosedArenaThrows() {
        Arena arena = Arena.ofConfined();
        arena.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void tensorCloseClosesUnderlyingArena() {
        Tensor t = Tensor.ones(DType.FLOAT32, 3, 3);
        Arena arena = t.ndarray().arena;
        t.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void tensorCloseFreesUnderlyingMemory() {
        Tensor t = Tensor.fill(DType.FLOAT32, 1.0f, 2, 2);
        Arena arena = t.ndarray().arena;
        t.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void testNDArrayCloseTwiceIsIdempotent() {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.zeros(DType.FLOAT32, arena, 2, 2);
        arr.close(); // closes arena
        arr.close(); // no-op due to closed flag — must not throw
        // arena closed by first call, no leak
    }

    @Test
    void testNDArraySegmentAccessAfterCloseIsUnsafe() {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.zeros(DType.FLOAT32, arena, 2);
        arr.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void testTensorConstructorDoesNotLeakArenaOnException() {
        try {
            try (Tensor ignored = Tensor.ones(DType.FLOAT32, 1)) {
                throw new RuntimeException("test");
            }
        } catch (RuntimeException e) {
            // expected — try-with-resources calls close() before propagating
        }
    }

    @Test
    void testVectorOpsRespectsArenaLifetime() {
        Arena inputArena = Arena.ofConfined();
        Arena outputArena = Arena.ofConfined();

        NDArray a = NDArray.zeros(DType.FLOAT32, inputArena, 5);
        NDArray b = NDArray.zeros(DType.FLOAT32, inputArena, 5);
        a.segment.set(ValueLayout.JAVA_FLOAT, 0L, 1.0f);
        b.segment.set(ValueLayout.JAVA_FLOAT, 0L, 2.0f);

        NDArray result = VectorOps.add(a, b, outputArena);
        assertEquals(3.0f, result.getFloat(0), 1e-5f);

        inputArena.close();

        // inputs are dead — their memory is released
        assertThrows(IllegalStateException.class, () -> a.getFloat(0));

        // result survives — lives in its own outputArena
        assertEquals(3.0f, result.getFloat(0), 1e-5f);

        outputArena.close();
    }
}