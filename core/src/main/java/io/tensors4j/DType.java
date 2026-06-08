package io.tensors4j;

import java.lang.foreign.ValueLayout;

// TODO: Implement UNALIGNED for all types
public enum DType {
    // Floating points
    FLOAT32(ValueLayout.JAVA_FLOAT, 4),
    FLOAT64(ValueLayout.JAVA_DOUBLE, 8),

    // Integers
    INT8(ValueLayout.JAVA_BYTE, 1),
    INT16(ValueLayout.JAVA_SHORT, 2),
    INT32(ValueLayout.JAVA_INT,   4),
    INT64(ValueLayout.JAVA_LONG,  8),

    // Boolean
    BOOL(ValueLayout.JAVA_BOOLEAN, 1);

    public final ValueLayout layout;
    public final int byteSize;

    DType(ValueLayout layout, int byteSize) {
        this.layout = layout;
        this.byteSize = byteSize;
    }
}