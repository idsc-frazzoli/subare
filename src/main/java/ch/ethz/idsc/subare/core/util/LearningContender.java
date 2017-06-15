// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.tensor.Scalar;

/**  */
public class LearningContender {
  public static LearningContender sarsa(MonteCarloInterface monteCarloInterface, Sarsa sarsa) {
    return new LearningContender(monteCarloInterface, sarsa, sarsa.qsa());
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DiscreteQsa qsa;
  private final DequeDigest dequeDigest;

  private LearningContender(MonteCarloInterface monteCarloInterface, DequeDigest sarsa, DiscreteQsa qsa) {
    this.monteCarloInterface = monteCarloInterface;
    this.qsa = qsa;
    this.dequeDigest = sarsa;
  }

  public Scalar stepAndCompare(Scalar epsilon, int nstep, DiscreteQsa ref) {
    PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa, epsilon);
    if (dequeDigest instanceof Sarsa) {
      Sarsa sarsa = (Sarsa) dequeDigest;
      sarsa.setPolicyInterface(policyInterface);
    }
    ExploringStarts.batch(monteCarloInterface, policyInterface, nstep, dequeDigest);
    return TensorValuesUtils.distance(qsa, ref);
  }
}
