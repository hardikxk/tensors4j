package io.tensors4j;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules", "jdk.incubator.vector"})
public class TensorBenchmark{

    @Param({"64", "512", "2048"})
    int size;

    @Benchmark
    public Object fillBench() {
        try (var t = Tensor.fill(DType.FLOAT32, 3.14f, size, size)) {
            return t;
        }
    }

    @Benchmark
    public float readBench() {
        try (var t = Tensor.ones(DType.FLOAT32, size, size)) {
            return t.getFloat(size/2, size/2);
        }
    }
}