// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.DoubleTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.SarsaEvaluationType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class DoubleTrueOnlineMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);
  private static final Scalar EPSILON = RealScalar.of(0.1);
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleTrueOnlineSarsa doubleTrueOnlineSarsa;

  // has convergence problems, don't use it yet!
  public DoubleTrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaEvaluationType evaluationType) {
    this.monteCarloInterface = monteCarloInterface;
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    LearningRate learningRate1 = ConstantLearningRate.of(ALPHA);
    LearningRate learningRate2 = ConstantLearningRate.of(ALPHA);
    doubleTrueOnlineSarsa = new DoubleTrueOnlineSarsa(monteCarloInterface, evaluationType, LAMBDA, learningRate1, learningRate2, featureMapper);
    doubleTrueOnlineSarsa.setExplore(EPSILON);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, doubleTrueOnlineSarsa.qsa(), EPSILON);
    ExploringStarts.batch(monteCarloInterface, policy, doubleTrueOnlineSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return doubleTrueOnlineSarsa.qsa();
  }
}
