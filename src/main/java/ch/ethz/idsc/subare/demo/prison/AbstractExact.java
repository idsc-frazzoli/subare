// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.util.LinkedList;
import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/* package */ abstract class AbstractExact {
  private final Supplier<Agent> sup1;
  private final Supplier<Agent> sup2;
  private final List<Opening> list = new LinkedList<>();

  class Opening {
    private final Agent a1 = sup1.get();
    private final Agent a2 = sup2.get();
    private final Scalar prob;
    private final Judger judger;

    public Opening(Integer[] a1open, Integer[] a2open) {
      a1.setOpeningSequence(a1open);
      a2.setOpeningSequence(a2open);
      int n = a1open.length + a2open.length;
      prob = RealScalar.of(1 << n).reciprocal();
      judger = new Judger(Arena.R0, a1, a2);
    }

    private void play(int epochs) {
      for (int round = 0; round < epochs; ++round)
        judger.play();
    }

    private Tensor exactRewards() {
      Tensor tensor = judger.ranking();
      // assert that no randomness was involved in the training
      if (a1.getRandomizedDecisionCount() != 0) {
        System.out.println(a1.getAbsDesc());
        System.out.println(SummaryString.of(a1));
        throw new IllegalStateException();
      }
      if (a2.getRandomizedDecisionCount() != 0) {
        System.out.println(a2.getAbsDesc());
        System.out.println(SummaryString.of(a2));
        throw new IllegalStateException();
      }
      return tensor.multiply(prob);
    }

    private Scalar actionReminder() {
      return RealScalar.of(a1.getActionReminder()).multiply(prob);
    }
  }

  public AbstractExact(Supplier<Agent> sup1, Supplier<Agent> sup2) {
    this.sup1 = sup1;
    this.sup2 = sup2;
  }

  public final void play(int epochs) {
    list.forEach(o -> o.play(epochs));
  }

  public final Tensor getExpectedRewards() {
    return list.stream() //
        .map(Opening::exactRewards) //
        .reduce(Tensor::add).get();
  }

  public final Scalar getActionReminder() {
    return list.stream() //
        .map(Opening::actionReminder) //
        .reduce(Scalar::add).get();
  }

  protected final void contribute(Integer[] a1open, Integer[] a2open) {
    list.add(new Opening(a1open, a2open));
  }
}
