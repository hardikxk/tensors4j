# Quick Demo
This demo showcases how to perform basic Tensor operations using Nadar.

```java
class Demo {
    static void main() {
        IO.println("--- Tensor Engine ---\n");

        // ALLOCATION — three factory methods
        try (
                Tensor weights = Tensor.fill(DType.FLOAT32, 0.5f, 1000, 1000);
                Tensor biases  = Tensor.ones(DType.FLOAT32, 1000, 1000);
                Tensor zeros   = Tensor.zeros(DType.FLOAT32, 1000, 1000)
        ) {
            IO.println("1. Allocation");
            IO.println("   weights : " + weights.shape()[0] + "x" + weights.shape()[1] + " filled with 0.5");
            IO.println("   biases  : " + biases.shape()[0]  + "x" + biases.shape()[1]  + " filled with 1.0");
            IO.println("   zeros   : " + zeros.shape()[0]   + "x" + zeros.shape()[1]   + " filled with 0.0");

            // ALLOCATING OPS — returns a new Tensor the caller owns
            IO.println("\n2. Allocating ops");
            try (Tensor preActivation = weights.add(biases)) {
                IO.println("   weights + biases[0,0]         = " + preActivation.getFloat(0, 0)); // 1.5

                // IN-PLACE OPS — mutate existing memory, zero allocation
                IO.println("\n3. In-place ops");
                preActivation.scale_(-1.0f);
                IO.println("   after scale_(-1.0)[0,0]       = " + preActivation.getFloat(0, 0)); // -1.5
                preActivation.relu_();
                IO.println("   after relu_()[0,0]             = " + preActivation.getFloat(0, 0)); // 0.0
            }

            // REDUCTIONS — return primitives, no allocation
            IO.println("\n4. Reductions");
            try (Tensor sampleData = Tensor.fill(DType.FLOAT32, 2.0f, 1000, 1000)) {
                IO.println("   sum of 1M x 2.0               = " + sampleData.sum());   // 2000000.0
                IO.println("   mean of 1M x 2.0              = " + sampleData.mean());  // 2.0
            }

            // ELEMENT ACCESS — get and set individual elements
            IO.println("\n5. Element access");
            try (Tensor t = Tensor.zeros(DType.FLOAT32, 4, 4)) {
                t.setFloat(99f, 2, 3);
                IO.println("   set [2,3]=99, get [2,3]       = " + t.getFloat(2, 3)); // 99.0
                IO.println("   untouched [0,0]               = " + t.getFloat(0, 0)); // 0.0
            }

            // ZERO-COPY VIEWS — transpose without copying data
            IO.println("\n6. Zero-copy transpose");
            try (Tensor mat = Tensor.fill(DType.FLOAT32, 0f, 3, 4)) {
                mat.setFloat(7f, 0, 3);
                NDArray transposed = mat.ndarray().transpose();
                IO.println("   mat[0,3]                      = " + mat.getFloat(0, 3));        // 7.0
                IO.println("   transposed[3,0]               = " + transposed.getFloat(3, 0)); // 7.0 same memory
                IO.println("   original shape                = " + mat.shape()[0] + "x" + mat.shape()[1]);
                IO.println("   transposed shape              = " + transposed.shape[0] + "x" + transposed.shape[1]);
            }

            // MATRIX MULTIPLICATION — tiled SIMD with FMA
            IO.println("\n7. Matrix multiplication");

            // Small matmul — verify correctness
            try (Tensor a = Tensor.fill(DType.FLOAT32, 1f, 2, 3);
                 Tensor b = Tensor.fill(DType.FLOAT32, 1f, 3, 4);
                 Tensor c = a.matmul(b)) {
                IO.println("   (2x3) @ (3x4) shape           = " + c.shape()[0] + "x" + c.shape()[1]);
                IO.println("   c[0,0] (should be 3.0)        = " + c.getFloat(0, 0));
            }

            // Medium matmul — exercises tiling path
            try (Tensor a = Tensor.fill(DType.FLOAT32, 1f, 128, 256);
                 Tensor b = Tensor.fill(DType.FLOAT32, 1f, 256, 128);
                 Tensor c = a.matmul(b)) {
                IO.println("   (128x256) @ (256x128) shape   = " + c.shape()[0] + "x" + c.shape()[1]);
                IO.println("   c[0,0] (should be 256.0)      = " + c.getFloat(0, 0));
            }

            // FORMATTED PRINT — toString output
            IO.println("\n8. Formatted output");
            try (Tensor small = Tensor.fill(DType.FLOAT32, 3.14f, 3, 3)) {
                IO.println(small.toString());
            }

        } // weights, biases, zeros freed here

        IO.println("--- Engine Shutdown Safely ---");
    }
}
```

## Output

```
--- Tensor Engine ---

1. Allocation
   weights : 1000x1000 filled with 0.5
   biases  : 1000x1000 filled with 1.0
   zeros   : 1000x1000 filled with 0.0

2. Allocating ops
   weights + biases[0,0]         = 1.5

3. In-place ops
   after scale_(-1.0)[0,0]       = -1.5
   after relu_()[0,0]             = 0.0

4. Reductions
   sum of 1M x 2.0               = 2000000.0
   mean of 1M x 2.0              = 2.0

5. Element access
   set [2,3]=99, get [2,3]       = 99.0
   untouched [0,0]               = 0.0

6. Zero-copy transpose
   mat[0,3]                      = 7.0
   transposed[3,0]               = 7.0
   original shape                = 3x4
   transposed shape              = 4x3

7. Matrix multiplication
   (2x3) @ (3x4) shape           = 2x4
   c[0,0] (should be 3.0)        = 3.0
   (128x256) @ (256x128) shape   = 128x128
   c[0,0] (should be 256.0)      = 256.0

8. Formatted output
NDArray[shape=[3, 3], dtype=FLOAT32]
[3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
--- Engine Shutdown Safely ---
```
