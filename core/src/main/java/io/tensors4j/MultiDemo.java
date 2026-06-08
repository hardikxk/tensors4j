package io.tensors4j;

public class MultiDemo {
    void main(){
        try(
                Tensor t1 = Tensor.fill(DType.FLOAT32, 2, 2, 3, 4, 5);
                Tensor t2 = t1;
                ){
            IO.println(t1.toString());
            t1.scale_(5);
            IO.println(t1.toString());
//            IO.println(t2.toString());

        }
    }
}
