package io.tensors4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * The core Array Datatype for Tensor objects and high dimensional numerical operations.
 * Includes standard methods for Tensor initialization
 * (in general this thing shouldn't be direct access rather only using the abstractions)
 * (but could potentially allow access to this as it can help make custom Tensor objects - right now leaving as is)
 */
public class NDArray implements AutoCloseable{
    // Core
    public final MemorySegment segment;
    final long offset;

    // Description - esk
    public final long[] shape;
    public final long[] strides;
    public final DType dtype;

    // The Arena
    public final Arena arena;
    private boolean closed = false;

    public NDArray(MemorySegment segment, long offset, long[] shape, long[] strides, DType dtype, Arena arena) {
        this.segment = segment;
        this.offset = offset;
        this.shape = shape;
        this.strides = strides;
        this.dtype = dtype;
        this.arena = arena;
    }

/** 2-D transpose. Returns a view — no data copied. Delegates to the general transpose. */
    public NDArray transpose() {
        return transpose(1, 0);
    }

    /**
     * General N-dimensional transpose. Returns a zero-copy view with axes permuted
     * according to the given permutation.
     *
     * @param axes a permutation of {0, 1, ..., n-1} specifying the new order of axes
     * @return a view with shape and strides permuted by {@code axes}
     * @throws IllegalArgumentException if {@code axes.length != shape.length},
     *                                  if any axis is out of range, or if there are duplicates
     */
    public NDArray transpose(int... axes) {
        java.util.Objects.requireNonNull(axes, "axes must not be null");
        if (axes.length != shape.length) {
            throw new IllegalArgumentException(
                "Expected " + shape.length + " axes, got " + axes.length);
        }
        boolean[] seen = new boolean[shape.length];
        for (int i = 0; i < axes.length; i++) {
            int a = axes[i];
            if (a < 0 || a >= shape.length) {
                throw new IllegalArgumentException(
                    "Axis " + a + " is out of range [0, " + (shape.length - 1) + "]");
            }
            if (seen[a]) {
                throw new IllegalArgumentException("Duplicate axis: " + a);
            }
            seen[a] = true;
        }
        long[] newShape   = new long[shape.length];
        long[] newStrides = new long[shape.length];
        for (int i = 0; i < axes.length; i++) {
            newShape[i]   = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }
        // arena = null: view does not own memory, close() is a no-op
        return new NDArray(this.segment, this.offset, newShape, newStrides, this.dtype, null);
    }

    /**
     * Returns a view of this array with a different logical shape, without copying data.
     * The array must be contiguous (row-major layout).
     * @param newShape the desired shape (element count must match current element count)
     * @return a view with the given shape and computed row-major strides
     * @throws UnsupportedOperationException if this array is not contiguous
     * @throws IllegalArgumentException      if the element counts do not match
     */
    public NDArray reshape(long... newShape) {
        java.util.Objects.requireNonNull(newShape, "newShape must not be null");
        for (long d : newShape) {
            if (d <= 0) {
                throw new IllegalArgumentException(
                    "Invalid reshape dimension: " + d + ". All dimensions must be positive.");
            }
        }
        if (!isContiguous()) {
            throw new UnsupportedOperationException(
                "reshape requires a contiguous (row-major) array");
        }
        if (elementCount(newShape) != elementCount(this.shape)) {
            throw new IllegalArgumentException(
                "New shape element count (" + elementCount(newShape)
                    + ") does not match current element count (" + elementCount(this.shape) + ")");
        }
        long[] newStrides = rowMajorStrides(newShape, this.dtype);
        return new NDArray(this.segment, this.offset, newShape, newStrides, this.dtype, null);
    }

    /**
     * Slices the first dimension. Returns a view no data copied.
     */
    public NDArray slice(long index) {
        if (index < 0 || index >= shape[0]) {
            throw new IndexOutOfBoundsException();
        }
        long   newOffset  = this.offset + (index * strides[0]);
        long[] newShape   = new long[shape.length - 1];
        long[] newStrides = new long[strides.length - 1];
        System.arraycopy(this.shape,   1, newShape,   0, newShape.length);
        System.arraycopy(this.strides, 1, newStrides, 0, newStrides.length);
        return new NDArray(this.segment, newOffset, newShape, newStrides, this.dtype, null);
    }

    // Helper to calculate the dimensions of strides
    private static long[] rowMajorStrides(long[] shape, DType dtype) {
        long[] strides = new long[shape.length];
        if(shape.length == 0){
            return strides;
        }
        strides[shape.length - 1] = dtype.byteSize;
        for (int i = shape.length - 2; i >= 0; i--)
            strides[i] = Math.multiplyExact(strides[i + 1], shape[i + 1]);
        return strides;
    }

    // This might be unnecessary. Remove this or the init method (probably the init method)
    public static NDArray zeros(DType dtype, Arena arena, long... shape) {
        return init(dtype, arena, shape); // allocate() already zeroes memory
    }

    public static NDArray ones(DType dtype, Arena arena, long... shape) {
        return fill(dtype, arena, 1.0f, shape); // allocate() already zeroes memory
    }

    // TODO: fix this... this is such a bad implementation even for a prototype!
    // Helper that returns the number of elements in the NDArray object based on shape.
    protected static long elementCount(long[] shape) {
        long n = 1;
        for (long d : shape){
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    // Helper to return the actual value
    public float getFloat(long... indices) {
        return segment.get(ValueLayout.JAVA_FLOAT, byteOffset(indices));
    }

    // Helper to set a value at the already allocated block
    public void setFloat(float value, long... indices) {
        segment.set(ValueLayout.JAVA_FLOAT, byteOffset(indices), value);
    }

    /**
     * Gets the float at a flat (linear) index into the logical element sequence.
     * NOTE: only correct for contiguous, non-offset arrays (i.e. freshly allocated,
     * not sliced or transposed). Use multi-index getFloat for views.
     */
    public float getFloatAtIndex(long flatIndex) {
        return segment.get(ValueLayout.JAVA_FLOAT, offset + flatIndex * dtype.byteSize);
    }

    /** Sets the float at a flat (linear) index. Same contiguity caveat as above. */
    public void setFloatAtIndex(float value, long flatIndex) {
        segment.set(ValueLayout.JAVA_FLOAT, offset + flatIndex * dtype.byteSize, value);
    }


    // Helper to calculate the offset
    private long byteOffset(long[] indices) {
        if(indices.length != shape.length){
            throw new IllegalArgumentException("Expected " + shape.length + " indices, got " + indices.length);
        }
        long off = offset;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for axis " + i + " with size " + shape[i]);
            }
            off += indices[i] * strides[i];
        }
        return off;
    }

    /**
     * @param value     The value to be assigned at each location of the array
     * @return      the array where each element is the value which was passed in the args.
     * Only supports assigning a scalar value simultaneously to all elements. (REFACTOR THIS ASAP!)
     */
    public static NDArray fill(DType dtype, Arena arena, float value, long... shape) {
        NDArray arr = init(dtype, arena, shape);
        if (dtype != DType.FLOAT32) {
            // fallback for non-float types
            long total = elementCount(shape);
            for (long i = 0; i < total; i++) arr.setFloatAtIndex(value, i);
            return arr;
        }
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        ByteOrder BO = ByteOrder.nativeOrder();
        long n = elementCount(shape);
        int step = SPECIES.length();
        long bound = n - (n % step);
        FloatVector vs = FloatVector.broadcast(SPECIES, value);
        long i = 0;
        while (i < bound) {
            vs.intoMemorySegment(arr.segment, (long) i * Float.BYTES, BO);
            i += step;
        }
        while (i < n) {
            arr.segment.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, value);
            i++;
        }
        return arr;
    }

    /**
     * @param dtype     The core (kinda primitive) type of object !Float32 by default!
     * @param arena     The assigned arena for the NDArray
     * @param shape     The dimensions of the tensor / array passed
     * @return      A tensor filled with zeroes
     */
    public static NDArray init(DType dtype, Arena arena, long... shape) {
        long total = elementCount(shape) * dtype.byteSize;
        MemorySegment seg = arena.allocate(total, dtype.byteSize);
        return new NDArray(seg, 0L, shape, rowMajorStrides(shape, dtype), dtype, arena);
    }

// THIS IS VERY EXPENSIVE (especially for `Arena.ofShared()` specifically) - need to think of something better (if that's even feasible)
    @Override
    public void close() {
        if (arena != null && !closed) {
            closed = true;
            arena.close();
        }
    }

    // TODO: Improve the printing here. make it pretty
    //  .
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NDArray[shape=").append(Arrays.toString(shape))
          .append(", dtype=").append(dtype).append("]\n");
        long total = elementCount(shape);
        sb.append("[");
        if (total <= 64) {
            for (long i = 0; i < total; i++) {
                sb.append(getFloatAtIndex(i));
                if (i < total - 1){
		            sb.append(", ");
		        }
            }
            sb.append("]");
        }
        else {
            for (long i = 0; i < 3; i++)
                sb.append(getFloatAtIndex(i)).append(", ");
            sb.append("..., ");
            for (long i = total - 3; i < total; i++) {
                sb.append(getFloatAtIndex(i));
                if (i < total - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
		}
        return sb.toString();
    }

    public boolean isContiguous() {
        long expectedStride = dtype.byteSize;

        for(int i = shape.length - 1; i >= 0; i--){
            if(strides[i] != expectedStride){
                return false;
            }
            expectedStride = Math.multiplyExact(expectedStride, shape[i]);
        }
        return true;
    }
}