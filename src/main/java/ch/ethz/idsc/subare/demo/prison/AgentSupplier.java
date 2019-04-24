// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.EGreedyAgent;
import ch.ethz.idsc.subare.ch02.GradientAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.RandomAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/* package */ enum AgentSupplier {
  ;
  public static final List<Supplier<Agent>> mixed = Arrays.asList( //
      // () -> new ConstantAgent(2, 0), //
      // () -> new ConstantAgent(2, 1), //
      () -> new TitForTatAgent(), //
      () -> new RandomAgent(2), //
      () -> new EGreedyAgent(2, i -> RationalScalar.of(1, 5), "1/5"), //
      () -> new EGreedyAgent(2, i -> RationalScalar.of(1, i.number().intValue() + 1), "1/i"), //
      () -> new GradientAgent(2, RealScalar.of(.1)), //
      () -> new OptimistAgent(2, RealScalar.of(6), RealScalar.of(.1)), //
      () -> new UCBAgent(2, RealScalar.of(1))) //
  ;

  public static List<Supplier<Agent>> getUCBs(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      list.add(() -> new UCBAgent(2, cs));
    }
    return list;
  }

  public static List<Supplier<Agent>> getOptimists(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      list.add(() -> new OptimistAgent(2, RealScalar.of(6), cs));
    }
    return list;
  }

  public static List<Supplier<Agent>> getEgreedyC(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      Supplier<Agent> sup = () -> new EGreedyAgent(2, i -> cs, cs.toString());
      list.add(sup);
    }
    return list;
  }
}
