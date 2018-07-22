// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/** uses 1-step digest */
public class SarsaMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar EPSILON = RealScalar.of(0.1);
  private static final int DIGEST_DEPTH = 1; // 0 is equal to the MonteCarlo approach
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final Sarsa sarsa;

  public SarsaMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    this.monteCarloInterface = monteCarloInterface;
    DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
    LearningRate learningRate = ConstantLearningRate.of(ALPHA);
    sarsa = sarsaType.supply(monteCarloInterface, qsaSarsa, learningRate);
    sarsa.setExplore(EPSILON);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), EPSILON);
    ExploringStarts.batch(monteCarloInterface, policy, DIGEST_DEPTH, sarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return sarsa.qsa();
  }
}
