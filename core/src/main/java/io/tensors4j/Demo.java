package io.tensors4j;

class Demo {
    public static void main(String[] args) {
        IO.println("--- Tensor Engine ---\n");

        // 1. ALLOCATION — three factory methods
        try (
                Tensor weights = Tensor.fill(DType.FLOAT32, 0.5f, 1000, 1000);
                Tensor biases  = Tensor.ones(DType.FLOAT32, 1000, 1000);
                Tensor zeros   = Tensor.zeros(DType.FLOAT32, 1000, 1000)
        ) {
            IO.println("1. Allocation");
            IO.println("   weights : " + weights.shape()[0] + "x" + weights.shape()[1] + " filled with 0.5");
            IO.println("   biases  : " + biases.shape()[0]  + "x" + biases.shape()[1]  + " filled with 1.0");
            IO.println("   zeros   : " + zeros.shape()[0]   + "x" + zeros.shape()[1]   + " filled with 0.0");

            // 2. ALLOCATING OPS — returns a new Tensor
            IO.println("\n2. Allocating ops");
            try (Tensor preActivation = weights.add(biases)) {
                IO.println("   weights + biases[0,0]         = " + preActivation.getFloat(0, 0)); // 1.5

                // 3. IN-PLACE OPS — mutate existing memory
                IO.println("\n3. In-place ops");
                preActivation.scale_(-1.0f);
                IO.println("   after scale_(-1.0)[0,0]       = " + preActivation.getFloat(0, 0)); // -1.5
                preActivation.relu_();
                IO.println("   after relu_()[0,0]             = " + preActivation.getFloat(0, 0)); // 0.0
            }

            // 4. ELEMENT-WISE (HADAMARD) PRODUCT
            IO.println("\n4. Element-wise product (mul)");
            try (Tensor a = Tensor.fill(DType.FLOAT32, 3.0f, 2, 3);
                 Tensor b = Tensor.fill(DType.FLOAT32, 2.0f, 2, 3);
                 Tensor c = a.mul(b)) {
                IO.println("   fill(3.0, 2x3) .* fill(2.0, 2x3) = c[0,0] " + c.getFloat(0, 0)); // 6.0
            }

            // 5. ALLOCATING SCALE
            IO.println("\n5. Allocating scale");
            try (Tensor t = Tensor.fill(DType.FLOAT32, 10.0f, 2, 2);
                 Tensor scaled = t.scale(0.5f)) {
                IO.println("   fill(10.0, 2x2).scale(0.5)   = " + scaled.getFloat(0, 0)); // 5.0
                IO.println("   original unchanged             = " + t.getFloat(0, 0));       // 10.0
            }

            // 6. REDUCTIONS
            IO.println("\n6. Reductions");
            try (Tensor sampleData = Tensor.fill(DType.FLOAT32, 2.0f, 1000, 1000)) {
                IO.println("   sum  of 1M x 2.0              = " + sampleData.sum());   // 2000000.0
                IO.println("   mean of 1M x 2.0              = " + sampleData.mean());  // 2.0
            }

            // 7. ELEMENT ACCESS
            IO.println("\n7. Element access");
            try (Tensor t = Tensor.zeros(DType.FLOAT32, 4, 4)) {
                t.setFloat(99f, 2, 3);
                IO.println("   set [2,3]=99, get [2,3]       = " + t.getFloat(2, 3)); // 99.0
                IO.println("   untouched [0,0]               = " + t.getFloat(0, 0)); // 0.0
            }

            // 8. ZERO-COPY TRANSPOSE
            IO.println("\n8. Zero-copy transpose");
            try (Tensor mat = Tensor.fill(DType.FLOAT32, 0f, 3, 4)) {
                mat.setFloat(7f, 0, 3);
                NDArray transposed = mat.ndarray().transpose();
                IO.println("   mat[0,3]                      = " + mat.getFloat(0, 3));        // 7.0
                IO.println("   transposed[3,0]               = " + transposed.getFloat(3, 0)); // 7.0 same memory
                IO.println("   original shape                = " + mat.shape()[0] + "x" + mat.shape()[1]);
                IO.println("   transposed shape              = " + transposed.shape[0] + "x" + transposed.shape[1]);
            }

            // 9. MATRIX MULTIPLICATION
            IO.println("\n9. Matrix multiplication (tiled SIMD + FMA)");
            try (Tensor a = Tensor.fill(DType.FLOAT32, 1f, 2, 3);
                 Tensor b = Tensor.fill(DType.FLOAT32, 1f, 3, 4);
                 Tensor c = a.matmul(b)) {
                IO.println("   (2x3) @ (3x4) shape           = " + c.shape()[0] + "x" + c.shape()[1]);
                IO.println("   c[0,0] (should be 3.0)        = " + c.getFloat(0, 0));
            }

            // 10. DOT PRODUCT — ND contraction
            IO.println("\n10. Dot product (ND contraction)");
            try (Tensor a = Tensor.fill(DType.FLOAT32, 1.0f, 2, 3, 4);
                 Tensor b = Tensor.fill(DType.FLOAT32, 1.0f, 5, 4, 6);
                 Tensor c = a.dot(b)) {
                IO.println("    (2,3,4) . (5,4,6) shape     = " + c.shape()[0] + "," + c.shape()[1] + "," + c.shape()[2] + "," + c.shape()[3]);
                IO.println("    c[0,0,0,0] (should be 4.0)   = " + c.getFloat(0, 0, 0, 0)); // 4.0
            }

            // 11. SUBTRACTION AND IN-PLACE ADDITION
            IO.println("\n11. Subtraction and in-place addition");
            try (Tensor a = Tensor.fill(DType.FLOAT32, 10.0f, 2, 3);
                 Tensor b = Tensor.fill(DType.FLOAT32, 3.0f, 2, 3);
                 Tensor c = a.sub(b)) {
                IO.println("    fill(10) - fill(3) = c[0,0]   = " + c.getFloat(0, 0)); // 7.0
                c.add_(b);
                IO.println("    after add_(b)[0,0]             = " + c.getFloat(0, 0)); // 10.0
            }

            // 12. TRANSPOSE, RESHAPE, SLICE
            IO.println("\n12. Transpose, reshape, slice");
            try (Tensor m = Tensor.fill(DType.FLOAT32, 0f, 3, 4)) {
                m.setFloat(7f, 0, 3);
                Tensor t = m.transpose();
                IO.println("    mat[0,3]=7, transposed[3,0]   = " + t.getFloat(3, 0)); // 7.0
                IO.println("    original shape                = " + m.shape()[0] + "x" + m.shape()[1]);
                IO.println("    transposed shape              = " + t.shape()[0] + "x" + t.shape()[1]);
                Tensor r = m.reshape(4, 3);
                IO.println("    reshape(4,3)[3,0]             = " + r.getFloat(3, 0)); // 7.0
                Tensor s = m.slice(0);
                IO.println("    slice(0).shape                = " + s.shape()[0] + " (was [3,4])");
                IO.println("    isContiguous (original)       = " + m.isContiguous());
            }

            // 13. FORMATTED OUTPUT
            IO.println("\n13. Formatted output");
            try (Tensor small = Tensor.fill(DType.FLOAT32, 3.14f, 3, 3)) {
                IO.println(small.toString());
            }

        } // all tensors freed here via try-with-resources

        IO.println("--- Engine Shutdown Safely --");
    }
}