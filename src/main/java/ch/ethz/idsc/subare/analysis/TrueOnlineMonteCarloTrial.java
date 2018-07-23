// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class TrueOnlineMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);
  private static final Scalar EPSILON = RealScalar.of(0.1);
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final TrueOnlineSarsa trueOnlineSarsa;

  public TrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    this.monteCarloInterface = monteCarloInterface;
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    LearningRate learningRate = ConstantLearningRate.of(ALPHA);
    trueOnlineSarsa = sarsaType.trueOnline(monteCarloInterface, LAMBDA, learningRate, featureMapper);
    trueOnlineSarsa.setExplore(EPSILON);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, trueOnlineSarsa.qsa(), EPSILON);
    ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return trueOnlineSarsa.qsa();
  }
}
