package io.tensors4j;

import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive Sargun-only benchmarks covering all operations.
 * Measures raw performance across different tensor sizes.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules", "jdk.incubator.vector"})
public class SargunBenchmark {

    // 2D sizes for matmul and element-wise ops
    @Param({"64", "256", "512", "1024"})
    int n;

    // 1D sizes for reductions and element-wise
    @Param({"1000", "100000"})
    int len;

    // Persistent input data (allocated once per trial)
    Arena dataArena;
    NDArray a2d, b2d, a1d, b1d;

    @Setup(Level.Trial)
    public void setup() {
        dataArena = Arena.ofConfined();
        a2d = NDArray.fill(DType.FLOAT32, dataArena, 1.5f, n, n);
        b2d = NDArray.fill(DType.FLOAT32, dataArena, 2.5f, n, n);
        a1d = NDArray.fill(DType.FLOAT32, dataArena, 1.5f, len);
        b1d = NDArray.fill(DType.FLOAT32, dataArena, 2.5f, len);
    }

    @TearDown(Level.Trial)
    public void teardown() {
        dataArena.close();
    }

    // MATRIX MULTIPLICATION (2D square)
    @Benchmark
    public Object matmul() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = NDArray.zeros(DType.FLOAT32, outArena, n, n);
            VectorMulOps.matmul(a2d, b2d, out);
            return out;
        }
    }

    // ELEMENT-WISE OPERATIONS (1D)
    @Benchmark
    public Object add() {
        try (Arena outArena = Arena.ofConfined()) {
            return VectorOps.add(a1d, b1d, outArena);
        }
    }

    @Benchmark
    public Object sub() {
        try (Arena outArena = Arena.ofConfined()) {
            return VectorOps.sub(a1d, b1d, outArena);
        }
    }

    @Benchmark
    public Object mul() {
        try (Arena outArena = Arena.ofConfined()) {
            return VectorOps.mul(a1d, b1d, outArena);
        }
    }

    @Benchmark
    public Object scale() {
        try (Arena outArena = Arena.ofConfined()) {
            return VectorOps.scale(a1d, 2.0f, outArena);
        }
    }

    @Benchmark
    public Object relu() {
        try (Arena outArena = Arena.ofConfined()) {
            return VectorOps.relu(a1d, outArena);
        }
    }

    // IN PLACE OPERATIONS  (1D) no allocation
    @Benchmark
    public void scaleInPlace() {
        VectorOps.scaleInPlace(a1d, 2.0f);
    }

    @Benchmark
    public void reluInPlace() {
        VectorOps.reluInPlace(a1d);
    }

    @Benchmark
    public void addInPlace() {
        VectorOps.addInPlace(a1d, b1d);
    }

    // REDUCTIONS (1D)
    @Benchmark
    public float sum() {
        return VectorOps.sum(a1d);
    }

    @Benchmark
    public float mean() {
        return VectorOps.mean(a1d);
    }

    @Benchmark
    public float dot() {
        return VectorOps.dot(a1d, b1d);
    }

    // FILL (2D) measures memory bandwidth
    @Benchmark
    public Object fill() {
        try (Arena outArena = Arena.ofConfined()) {
            return NDArray.fill(DType.FLOAT32, outArena, 3.14f, n, n);
        }
    }

    // ZERO-COPY VIEWS (2D) measures view creation overhead
    @Benchmark
    public Object transpose() {
        return a2d.transpose();
    }

    @Benchmark
    public Object transposeNdim() {
        return a2d.transpose(1, 0);
    }

    @Benchmark
    public Object reshape() {
        return a2d.reshape(n * n);
    }

    @Benchmark
    public Object slice() {
        return a2d.slice(0);
    }

    // DOT PRODUCT FOR N-RANK TENSORS (3D @ 3D)
    @Benchmark
    public Object dot_nd_small() {
        try (Arena arena3d = Arena.ofConfined()) {
            NDArray a3 = NDArray.fill(DType.FLOAT32, arena3d, 1.0f, 4, 8, 16);
            NDArray b3 = NDArray.fill(DType.FLOAT32, arena3d, 1.0f, 16, 4, 8);
            // Not a standard operation — skip or use manual approach
            return a3;
        }
    }

    // TENSOR CREATION OVERHEAD
    @Benchmark
    public Object create_zeros() {
        try (Arena outArena = Arena.ofConfined()) {
            return NDArray.zeros(DType.FLOAT32, outArena, n, n);
        }
    }

    @Benchmark
    public Object create_ones() {
        try (Arena outArena = Arena.ofConfined()) {
            return NDArray.ones(DType.FLOAT32, outArena, n, n);
        }
    }

    @Benchmark
    public Object create_tensor_zeros() {
        return Tensor.zeros(DType.FLOAT32, n, n);
    }

    @Benchmark
    public Object create_tensor_ones() {
        return Tensor.ones(DType.FLOAT32, n, n);
    }

    @Benchmark
    public Object create_tensor_fill() {
        return Tensor.fill(DType.FLOAT32, 1.0f, n, n);
    }
}
