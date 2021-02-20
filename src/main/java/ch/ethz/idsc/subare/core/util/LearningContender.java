// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;

/**  */
public class LearningContender {
  /** @param monteCarloInterface
   * @param sarsa
   * @return */
  public static LearningContender sarsa(MonteCarloInterface monteCarloInterface, Sarsa sarsa) {
    return new LearningContender(monteCarloInterface, sarsa, sarsa.qsa());
  }

  /***************************************************/
  private final MonteCarloInterface monteCarloInterface;
  private final DiscreteQsa qsa;
  private final Sarsa sarsa;

  private LearningContender(MonteCarloInterface monteCarloInterface, Sarsa sarsa, DiscreteQsa qsa) {
    this.monteCarloInterface = monteCarloInterface;
    this.qsa = qsa;
    this.sarsa = sarsa;
  }

  public void stepAndCompare(ExplorationRate explorationRate, int nstep, DiscreteQsa ref) {
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, ref, sarsa.sac());
    policy.setExplorationRate(explorationRate);
    ExploringStarts.batch(monteCarloInterface, policy, nstep, sarsa);
  }

  public Infoline infoline(DiscreteQsa ref) {
    return new Infoline(monteCarloInterface, ref, qsa);
  }
}
