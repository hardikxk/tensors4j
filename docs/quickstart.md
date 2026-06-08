# Quickstart

Here is how to allocate your first tensor:

```java
try (Tensor a = Tensor.ones(DType.FLOAT32, 10, 10)) {
    System.out.println(a);
}

