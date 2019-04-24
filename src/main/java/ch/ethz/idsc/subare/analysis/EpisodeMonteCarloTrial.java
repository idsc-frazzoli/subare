// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.PolicyBase;

/* package */ class EpisodeMonteCarloTrial implements MonteCarloTrial {
  private final MonteCarloInterface monteCarloInterface;
  private final MonteCarloExploringStarts mces;
  private final PolicyBase policy;

  public EpisodeMonteCarloTrial(MonteCarloInterface monteCarloInterface, PolicyBase policy) {
    this.monteCarloInterface = monteCarloInterface;
    this.mces = new MonteCarloExploringStarts(monteCarloInterface);
    this.policy = policy;
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, policy, mces);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return mces.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepInterface stepInterface) {
    throw new UnsupportedOperationException();
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }
}
