package io.tensors4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.util.concurrent.TimeUnit;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Head-to-head benchmarks: Sargun vs ND4J.
 *
 * ND4J is the most widely used Java tensor library (backed by Deep Java Library).
 * It uses JNI to native BLAS (OpenBLAS/MKL) for compute.
 *
 * Sargun uses pure Java 25 Panama FFM + Vector API no JNI, no native code.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules", "jdk.incubator.vector"})
public class SargunVsND4JBenchmark {

    // ── 2D sizes for matmul ──
    @Param({"64", "256", "512", "1024"})
    int n;

    // ── 1D sizes for element-wise and reductions ──
    @Param({"1000", "10000", "100000"})
    int len;

    // ── ND4J persistent state (allocated once per trial) ──
    INDArray nd4jA2d, nd4jB2d, nd4jA1d, nd4jB1d;

    // ── Sargun persistent state (allocated once per trial) ──
    Arena dataArena;
    NDArray sargunA2d, sargunB2d, sargunA1d, sargunB1d;

    @Setup(Level.Trial)
    public void setup() {
        // ND4J: let ND4J manage its own memory
        nd4jA2d = Nd4j.rand(DataType.FLOAT, n, n);
        nd4jB2d = Nd4j.rand(DataType.FLOAT, n, n);
        nd4jA1d = Nd4j.rand(DataType.FLOAT, len);
        nd4jB1d = Nd4j.rand(DataType.FLOAT, len);

        // Sargun: allocate once, reuse across benchmark iterations
        dataArena = Arena.ofConfined();
        sargunA2d = NDArray.fill(DType.FLOAT32, dataArena, 1.5f, n, n);
        sargunB2d = NDArray.fill(DType.FLOAT32, dataArena, 2.5f, n, n);
        sargunA1d = NDArray.fill(DType.FLOAT32, dataArena, 1.5f, len);
        sargunB1d = NDArray.fill(DType.FLOAT32, dataArena, 2.5f, len);
    }

    @TearDown(Level.Trial)
    public void teardown() {
        dataArena.close();
        // ND4J memory is GC-managed
    }

    // MATRIX MULTIPLICATION (2D)

    @Benchmark
    public Object matmul_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = NDArray.zeros(DType.FLOAT32, outArena, n, n);
            VectorMulOps.matmul(sargunA2d, sargunB2d, out);
            return out;
        }
    }

    @Benchmark
    public Object matmul_nd4j() {
        return nd4jA2d.mmul(nd4jB2d);
    }

    // ELEMENT-WISE ADDITION (1D)
    @Benchmark
    public Object add_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = VectorOps.add(sargunA1d, sargunB1d, outArena);
            return out;
        }
    }

    @Benchmark
    public Object add_nd4j() {
        return nd4jA1d.add(nd4jB1d);
    }

    // ELEMENT-WISE MULTIPLICATION / HADAMARD PRODUCT (1D)
    @Benchmark
    public Object mul_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = VectorOps.mul(sargunA1d, sargunB1d, outArena);
            return out;
        }
    }

    @Benchmark
    public Object mul_nd4j() {
        return nd4jA1d.mul(nd4jB1d);
    }

    // SCALE (multiply by scalar) (1D)
    @Benchmark
    public Object scale_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = VectorOps.scale(sargunA1d, 2.0f, outArena);
            return out;
        }
    }

    @Benchmark
    public Object scale_nd4j() {
        return nd4jA1d.mul(2.0f);
    }

    // ReLU (1D)
    @Benchmark
    public Object relu_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            NDArray out = VectorOps.relu(sargunA1d, outArena);
            return out;
        }
    }

    @Benchmark
    public Object relu_nd4j() {
        return Transforms.relu(nd4jA1d, false);
    }

    // SUM REDUCTION (1D)
    @Benchmark
    public float sum_sargun() {
        return VectorOps.sum(sargunA1d);
    }

    @Benchmark
    public float sum_nd4j() {
        return nd4jA1d.sumNumber().floatValue();
    }

    // DOT PRODUCT (1D)
    @Benchmark
    public float dot_sargun() {
        return VectorOps.dot(sargunA1d, sargunB1d);
    }

    @Benchmark
    public float dot_nd4j() {
        return (float) nd4jA1d.mul(nd4jB1d).getDouble(0);
    }

    // TRANSPOSE (2D)
    @Benchmark
    public Object transpose_sargun() {
        return sargunA2d.transpose();
    }

    @Benchmark
    public Object transpose_nd4j() {
        return nd4jA2d.transpose();
    }

    // FILL (2D)
    @Benchmark
    public Object fill_sargun() {
        try (Arena outArena = Arena.ofConfined()) {
            return NDArray.fill(DType.FLOAT32, outArena, 3.14f, n, n);
        }
    }

    @Benchmark
    public Object fill_nd4j() {
        return Nd4j.rand(DataType.FLOAT, n, n).mul(3.14);
    }
}
