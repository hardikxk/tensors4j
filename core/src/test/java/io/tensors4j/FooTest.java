package io.tensors4j;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FooTest {

    @Test
    void memWriteAndRead(){
        try(Arena arena = Arena.ofConfined()) {
            MemorySegment memorySegment = arena.allocate(ValueLayout.JAVA_INT);

            memorySegment.set(ValueLayout.JAVA_INT, 0L, 11);

            var value = memorySegment.get(ValueLayout.JAVA_INT, 0L);

            assertEquals(11, value);
        }
    }

    @Test
    void throwIllegalStateExceptionError(){
        MemorySegment memorySegment;
        try(Arena arena = Arena.ofConfined()) {
            memorySegment = arena.allocate(ValueLayout.JAVA_INT);

            memorySegment.set(ValueLayout.JAVA_INT, 0L, 11);

        }

        assertThrows(IllegalStateException.class, () -> {
            memorySegment.get(ValueLayout.JAVA_INT, 0L);
        });
    }
}