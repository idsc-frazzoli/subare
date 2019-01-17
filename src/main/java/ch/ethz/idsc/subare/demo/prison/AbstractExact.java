// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;

abstract class AbstractExact {
  final Supplier<Agent> sup1;
  final Supplier<Agent> sup2;
  final int epochs;
  Tensor expected = Array.zeros(2);

  public AbstractExact(Supplier<Agent> sup1, Supplier<Agent> sup2, int epochs) {
    this.sup1 = sup1;
    this.sup2 = sup2;
    this.epochs = epochs;
  }

  public final Tensor getExpectedRewards() {
    return expected.copy();
  }

  protected final void contribute(Integer[] a1open, Integer[] a2open) {
    int n = a1open.length + a2open.length;
    Scalar prob = RealScalar.of(1 << n).reciprocal();
    Tensor rew = exactRewards(a1open, a2open);
    expected = expected.add(rew.multiply(prob));
  }

  private Tensor exactRewards(Integer[] a1open, Integer[] a2open) {
    Agent a1 = sup1.get();
    a1.setOpeningSequence(a1open);
    Agent a2 = sup2.get();
    a2.setOpeningSequence(a2open);
    Tensor tensor = Training.train(a1, a2, epochs);
    // assert that no randomness was involved in the training
    GlobalAssert.that(a1.getRandomizedDecisionCount() == 0);
    if (a2.getRandomizedDecisionCount() != 0) {
      System.out.println(SummaryString.of(a2));
    }
    GlobalAssert.that(a2.getRandomizedDecisionCount() == 0);
    return tensor;
  }
}
