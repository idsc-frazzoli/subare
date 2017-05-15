// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.io.File;
import java.io.IOException;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.subare.core.Settings;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.io.Put;

class UCBsExact extends AbstractExact {
  public UCBsExact(Supplier<Agent> sup1, Supplier<Agent> sup2, int epochs) {
    super(sup1, sup2, epochs);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }

  public static void showOne() {
    Supplier<Agent> sup1 = //
        () -> new UCBAgent(2, RationalScalar.of(10, 10));
    Supplier<Agent> sup2 = //
        () -> new UCBAgent(2, RationalScalar.of(8, 10));
    UCBsExact exact = new UCBsExact(sup1, sup2, 200);
    System.out.println(exact.getExpectedRewards());
  }

  public static void main(String[] args) throws IOException {
    Tensor init = Tensors.vector(i -> RationalScalar.of(40 + i, 80), 80);
    Tensor res = Tensors.empty();
    for (Tensor c0 : init) {
      Tensor row = Tensors.empty();
      for (Tensor c1 : init) {
        Supplier<Agent> sup1 = //
            () -> new UCBAgent(2, (Scalar) c0);
        Supplier<Agent> sup2 = //
            () -> new UCBAgent(2, (Scalar) c1);
        UCBsExact exact = new UCBsExact(sup1, sup2, 50);
        row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
      }
      res.append(row);
    }
    Put.of(new File(Settings.root(), "ucb"), res);
  }
}
