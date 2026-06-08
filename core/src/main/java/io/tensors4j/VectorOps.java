// This is not perfect. IK. I lost my mind here.
// This serves as the Base to build upon and has the basic SIMD implementation using the VectorAPI.

package io.tensors4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

// TODO: rewrite the docstring here (and cleanup unused commented code)
/**
 */
public final class VectorOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder BO = ByteOrder.nativeOrder();

    private VectorOps() {
    }

    //  Core helpers
    //
    // applyUnary / applyBinary are private static with a fixed call site per
    // caller (relu always passes the same lambda, add always passes the same
    // lambda). The JIT inlines applyUnary into the caller, then inlines the
    // lambda, producing identical assembly to a hardcoded loop. No megamorphic
    // dispatch — each call site is monomorphic.

//    @FunctionalInterface
//    private interface VecUnaryOp {
//        FloatVector apply(FloatVector v);
//    }
//
//    @FunctionalInterface
//    private interface VecBinaryOp {
//        FloatVector apply(FloatVector a, FloatVector b);
//    }
//
//    @FunctionalInterface
//    private interface ScalarUnaryOp {
//        float apply(float v);
//    }
//
//    @FunctionalInterface
//    private interface ScalarBinaryOp {
//        float apply(float a, float b);
//    }

//    /**
//     * <p>
//     * Perform SIMD operations by offloading a calculated amount of values and performing the operation on all of them.
//     * </p>
//     * Reads from src, applies vOp (SIMD) / sOp (scalar tail), writes to dst.
//     * src == dst is valid for in-place ops.
//     */
//    private static void applyUnary(NDArray src, NDArray dst,
//                                   VecUnaryOp vOp, ScalarUnaryOp sOp) {
//        int n = (int) NDArray.elementCount(src.shape);  // Total elements
//        int bound = SPECIES.loopBound(n);   // it's like the last element that can fit according to the step. (the bound basically)
//        int step = SPECIES.length();    // Floats in a single clock cycle
//        int i = 0;
//
//        while (i < bound) {
//            long bo = (long) i * Float.BYTES;
//            vOp.apply(FloatVector.fromMemorySegment(SPECIES, src.segment, src.offset + bo, BO))
//                    .intoMemorySegment(dst.segment, dst.offset + bo, BO);
//            i += step;
//        }
//
//        // for the leftover elements from the first pass.
//        while (i < n) {
//            long bo = (long) i * Float.BYTES;
//            dst.segment.set(ValueLayout.JAVA_FLOAT, dst.offset + bo,
//                    sOp.apply(src.segment.get(ValueLayout.JAVA_FLOAT, src.offset + bo)));
//            i++;
//        }
//    }
//
//    private static void applyBinary(NDArray a, NDArray b, NDArray dst,
//                                    VecBinaryOp vOp, ScalarBinaryOp sOp) {
//        int n = (int) NDArray.elementCount(a.shape);
//        int bound = SPECIES.loopBound(n);
//        int step = SPECIES.length();
//        int i = 0;
//
//        while (i < bound) {
//            long bo = (long) i * Float.BYTES;
//            vOp.apply(
//                    FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO),
//                    FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO)
//            ).intoMemorySegment(dst.segment, dst.offset + bo, BO);
//            i += step;
//        }
//        while (i < n) {
//            long bo = (long) i * Float.BYTES;
//            dst.segment.set(ValueLayout.JAVA_FLOAT, dst.offset + bo,
//                    sOp.apply(
//                            a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo),
//                            b.segment.get(ValueLayout.JAVA_FLOAT, b.offset + bo)));
//            i++;
//        }
//    }

    //  Allocating binary ops
    // TODO: fix the copy pasted dup code into a helper method. (housekeeping!)
    /**
     * out = a + b
     */
    public static NDArray add(NDArray a, NDArray b, Arena arena) {
        checkBinary(a, b);
        NDArray out = NDArray.zeros(a.dtype, arena, a.shape);

        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO);
            va.add(vb).intoMemorySegment(out.segment, out.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            out.segment.set(ValueLayout.JAVA_FLOAT, out.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) +
                            b.segment.get(ValueLayout.JAVA_FLOAT, b.offset + bo));
            i++;
        }
        return out;
    }

    /**
     * out = a - b
     */
    public static NDArray sub(NDArray a, NDArray b, Arena arena) {
        checkBinary(a, b);
        NDArray out = NDArray.zeros(a.dtype, arena, a.shape);

        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO);
            va.sub(vb).intoMemorySegment(out.segment, out.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            out.segment.set(ValueLayout.JAVA_FLOAT, out.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) -
                            b.segment.get(ValueLayout.JAVA_FLOAT, b.offset + bo));
            i++;
        }
        return out;
    }

    /**
     * out = a * b  element-wise
     * Hadamard product
     */
    public static NDArray mul(NDArray a, NDArray b, Arena arena) {
        checkBinary(a, b);
        NDArray out = NDArray.zeros(a.dtype, arena, a.shape);

        int n = (int) NDArray.elementCount(a.shape);
        int bound = VectorOps.SPECIES.loopBound(n);
        int step = VectorOps.SPECIES.length();
        int i = 0;

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(VectorOps.SPECIES, a.segment, a.offset + bo, VectorOps.BO);
            FloatVector vb = FloatVector.fromMemorySegment(VectorOps.SPECIES, b.segment, b.offset + bo, VectorOps.BO);
            va.mul(vb).intoMemorySegment(out.segment, out.offset + bo, VectorOps.BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            out.segment.set(ValueLayout.JAVA_FLOAT, out.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) *
                            b.segment.get(ValueLayout.JAVA_FLOAT, b.offset + bo));
            i++;
        }
        return out;
    }

    /**
     * out = a * scalar
     */
    public static NDArray scale(NDArray a, float scalar, Arena arena) {
        checkUnary(a);
        NDArray out = NDArray.zeros(a.dtype, arena, a.shape);

        int n = (int) NDArray.elementCount(a.shape);
        int bound = VectorOps.SPECIES.loopBound(n);
        int step = VectorOps.SPECIES.length();
        int i = 0;

        FloatVector vs = FloatVector.broadcast(VectorOps.SPECIES, scalar);

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(VectorOps.SPECIES, a.segment, a.offset + bo, VectorOps.BO)
                    .mul(vs)
                    .intoMemorySegment(out.segment, out.offset + bo, VectorOps.BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            out.segment.set(ValueLayout.JAVA_FLOAT, out.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) * scalar);
            i++;
        }
        return out;
    }

    /**
     * out = max(0, a)
     */
    public static NDArray relu(NDArray a, Arena arena) {
        checkUnary(a);
        NDArray out = NDArray.zeros(a.dtype, arena, a.shape);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        FloatVector vz = FloatVector.zero(SPECIES);

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO)
                    .max(vz)
                    .intoMemorySegment(out.segment, out.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            out.segment.set(ValueLayout.JAVA_FLOAT, out.offset + bo,
                    Math.max(0f, a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo)));
            i++;
        }
        return out;
    }

    //  In place ops

    /**
     * a[i] += b[i]
     */
    public static void addInPlace(NDArray a, NDArray b) {
        checkBinary(a, b);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO);
            va.add(vb).intoMemorySegment(a.segment, a.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            a.segment.set(ValueLayout.JAVA_FLOAT, a.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) +
                            b.segment.get(ValueLayout.JAVA_FLOAT, b.offset + bo));
            i++;
        }
    }

    /**
     * a[i] *= scalar
     */
    public static void scaleInPlace(NDArray a, float scalar) {
        checkUnary(a);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        FloatVector vs = FloatVector.broadcast(SPECIES, scalar);

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO)
                    .mul(vs)
                    .intoMemorySegment(a.segment, a.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            a.segment.set(ValueLayout.JAVA_FLOAT, a.offset + bo,
                    a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo) * scalar);
            i++;
        }
    }

    /**
     * a[i] = max(0, a[i])
     */
    public static void reluInPlace(NDArray a) {
        checkUnary(a);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        int i = 0;

        FloatVector vz = FloatVector.zero(SPECIES);

        while (i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO)
                    .max(vz)
                    .intoMemorySegment(a.segment, a.offset + bo, BO);
            i += step;
        }
        while (i < n) {
            long bo = (long) i * Float.BYTES;
            a.segment.set(ValueLayout.JAVA_FLOAT, a.offset + bo,
                    Math.max(0f, a.segment.get(ValueLayout.JAVA_FLOAT, a.offset + bo)));
            i++;
        }
    }

    // Pattern: accumulate in SIMD across the whole loop (stay in SIMD registers),
    // call reduceLanes exactly once outside the loop, then handle the tail with
    // a masked load so inactive lanes contribute 0 to the reduction automatically.
    /**
     * Sum of all elements.
     */
    public static float sum(NDArray a) {
        checkUnary(a);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();

        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;

        while(i < bound) {
            acc = acc.add(FloatVector.fromMemorySegment(
                    SPECIES, a.segment, a.offset + (long) i * Float.BYTES, BO));
            i += step;
        }

        // One scalar crossing for the full accumulator
        float total = acc.reduceLanes(VectorOperators.ADD);

        // Masked tail — inactive lanes are 0, safe to reduce without branching
        if (i < n) {
            VectorMask<Float> mask = SPECIES.indexInRange(i, n);
            total += FloatVector.fromMemorySegment(
                            SPECIES, a.segment, a.offset + (long) i * Float.BYTES, BO, mask)
                    .reduceLanes(VectorOperators.ADD, mask);
        }

        return total;
    }

    /**
     * Arithmetic mean.
     */
    public static float mean(NDArray a) {
        return sum(a) / NDArray.elementCount(a.shape);
    }

    /**
     * Dot product of two flat contiguous FLOAT32 arrays.
     * Uses FMA: acc += va * vb in one hardware instruction per lane group.
     * Building block for BlasEngine matmul.
     */
    public static float dot(NDArray a, NDArray b) {
        checkBinary(a, b);
        int n = (int) NDArray.elementCount(a.shape);
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();

        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;

        while(i < bound) {
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO);
            acc = va.fma(vb, acc);  // acc += va * vb, single hardware FMA instruction
            i += step;
        }

        float total = acc.reduceLanes(VectorOperators.ADD);

        if (i < n) {
            VectorMask<Float> mask = SPECIES.indexInRange(i, n);
            long bo = (long) i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a.segment, a.offset + bo, BO, mask);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, b.offset + bo, BO, mask);
            total += va.mul(vb).reduceLanes(VectorOperators.ADD, mask);
            // Note: no fma for tail — fma requires full vector width to be meaningful
        }

        return total;
    }

    // Helpers

    private static void checkUnary(NDArray a) {
        if (a.dtype != DType.FLOAT32)
            throw new IllegalArgumentException("VectorOps requires FLOAT32, got " + a.dtype);
        if (!a.isContiguous())
            throw new UnsupportedOperationException(
                    "Non-contiguous arrays not supported. Call .compact() first.");
    }

    private static void checkBinary(NDArray a, NDArray b) {
        checkUnary(a);
        checkUnary(b);
        if (!Arrays.equals(a.shape, b.shape))
            throw new IllegalArgumentException("Shape mismatch: "
                    + Arrays.toString(a.shape) + " vs " + Arrays.toString(b.shape));
    }
}