// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;

public class DoubleSarsaMonteCarloTrial implements MonteCarloTrial {
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleSarsa doubleSarsa;

  public DoubleSarsaMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    this.monteCarloInterface = monteCarloInterface;
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    doubleSarsa = new DoubleSarsa(sarsaType, monteCarloInterface, //
        qsa1, qsa2, //
        DefaultLearningRate.of(1, .51), //
        DefaultLearningRate.of(1, .51));
    doubleSarsa.setExplore(RealScalar.of(0.1));
  }

  public void executeBatch() {
    Policy policy = doubleSarsa.getEGreedy();
    ExploringStarts.batch(monteCarloInterface, policy, 1, doubleSarsa);
  }

  @Override
  public DiscreteQsa qsa() {
    return doubleSarsa.qsa();
  }
}
