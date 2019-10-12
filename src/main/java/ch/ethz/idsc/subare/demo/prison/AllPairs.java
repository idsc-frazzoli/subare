// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.sca.Chop;

/* package */ enum AllPairs {
  ;
  static Tensor performance(List<Supplier<Agent>> list, int runs, int epochs) {
    final int size = list.size();
    Tensor matrix = Array.zeros(size, size);
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
        Tensor table = Tensors.empty();
        for (int run = 0; run < runs; ++run) {
          Agent a1 = list.get(i1).get();
          Agent a2 = list.get(i2).get();
          table.append(Training.train(a1, a2, epochs));
        }
        if (table.length() != runs)
          throw new RuntimeException();
        Tensor mean = Mean.of(table);
        Chop.NONE.requireAllZero(matrix.Get(i1, i2));
        Chop.NONE.requireAllZero(matrix.Get(i2, i1));
        matrix.set(mean.Get(0), i1, i2);
        matrix.set(mean.Get(1), i2, i1);
      }
    }
    return matrix;
  }
}
