package io.tensors4j;

import org.openjdk.jmh.annotations.*;

        import java.lang.foreign.Arena;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector"
})
public class TempVectorOpsBenchmark {

    @Param({"256", "4096", "1048576"})
    int size;

    Arena dataArena;
    NDArray a, b;

    @Setup(Level.Trial)
    public void setup() {
        // This arena ONLY holds the input data. It does not grow.
        dataArena = Arena.ofConfined();
        a = NDArray.fill(DType.FLOAT32, dataArena, 1.5f, size);
        b = NDArray.fill(DType.FLOAT32, dataArena, 2.5f, size);
    }

    @TearDown(Level.Trial)
    public void teardown() {
        dataArena.close();
    }

    // This is safe because it just overwrites existing memory.
    @Benchmark
    public void scale_in_place_helper() {
        VectorOps.scaleInPlace(a, 2.0f);
    }

    @Benchmark
    public void sum() {
        VectorOps.sum(a); // Reductions return a primitive float, no memory leak
    }

    @Benchmark
    public void dot() {
        VectorOps.dot(a, b); // Returns primitive float, no memory leak
    }

    @Benchmark
    public float add_allocating() {
        // Create a local arena that dies at the end of the loop!
        try (Arena localArena = Arena.ofConfined()) {
            NDArray result = VectorOps.add(a, b, localArena);
            // Return a single float to prevent the JVM from optimizing the math away
            return result.getFloat(0);
        }
    }

    @Benchmark
    public float scale_helper_allocating() {
        try (Arena localArena = Arena.ofConfined()) {
            NDArray result = VectorOps.scale(a, 2.0f, localArena);
            return result.getFloat(0);
        }
    }

    @Benchmark
    public float scale_flattened_allocating() {
        try (Arena localArena = Arena.ofConfined()) {
            NDArray out = NDArray.zeros(DType.FLOAT32, localArena, a.shape);

            // TODO: .....
            return out.getFloat(0);
        }
    }
}