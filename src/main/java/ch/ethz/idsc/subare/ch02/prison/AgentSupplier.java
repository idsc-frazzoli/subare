package ch.ethz.idsc.subare.ch02.prison;

import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.ConstantAgent;
import ch.ethz.idsc.subare.ch02.EGreedyAgent;
import ch.ethz.idsc.subare.ch02.GradientAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.RandomAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;

public enum AgentSupplier {
  agent0(() -> new ConstantAgent(0)), //
  agent1(() -> new ConstantAgent(1)), //
  random(() -> new RandomAgent(2)), //
  egreec(() -> new EGreedyAgent(2, i -> RationalScalar.of(1, 5), "1/5")), //
  egreed(() -> new EGreedyAgent(2, i -> RationalScalar.of(1, i + 1), "1/i")), //
  gradie(() -> new GradientAgent(2, RealScalar.of(.1))), //
  optimi(() -> new OptimistAgent(2, RealScalar.of(3), RealScalar.of(.1))), //
  ucboun(() -> new UCBAgent(2, RealScalar.of(1))), //
  ;
  public final Supplier<Agent> supplier;

  AgentSupplier(Supplier<Agent> supplier) {
    this.supplier = supplier;
  }
}
