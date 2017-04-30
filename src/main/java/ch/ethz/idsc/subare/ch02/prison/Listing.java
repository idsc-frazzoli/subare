// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Tensor;

class Listing {
  public static void main(String[] args) {
    List<Supplier<Agent>> list = AgentSupplier.mixed;
    Tensor matrix = AllPairs.performance(list, 20, 100);
    final int size = matrix.length();
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
        Agent a1 = list.get(i1).get();
        Agent a2 = list.get(i2).get();
        System.out.println("---");
        System.out.println(String.format("%s %f", //
            a1.getAbsDesc(), matrix.Get(i1, i2).number().doubleValue()));
        System.out.println(String.format("%s %f", //
            a2.getAbsDesc(), matrix.Get(i2, i1).number().doubleValue()));
      }
    }
    System.out.println("done.");
    int asd = Integer.parseInt("000123");
    System.out.println(asd);
  }
}
