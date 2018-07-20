// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class EpisodeMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar EPSILON = RealScalar.of(0.1);
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final MonteCarloExploringStarts monteCarloExploringStarts;

  public EpisodeMonteCarloTrial(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
    monteCarloExploringStarts = new MonteCarloExploringStarts(monteCarloInterface);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, monteCarloExploringStarts.qsa(), EPSILON);
    ExploringStarts.batch(monteCarloInterface, policy, monteCarloExploringStarts);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return monteCarloExploringStarts.qsa();
  }
}
