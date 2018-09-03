// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class DoubleTrueOnlineMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);
  private static final Scalar EPSILON = RealScalar.of(0.1);

  public static DoubleTrueOnlineMonteCarloTrial create(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    return new DoubleTrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, //
        featureMapper, //
        ConstantLearningRate.of(ALPHA), ConstantLearningRate.of(ALPHA), //
        new FeatureWeight(featureMapper), new FeatureWeight(featureMapper));
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleTrueOnlineSarsa doubleTrueOnlineSarsa;

  // has convergence problems, don't use it yet!
  private DoubleTrueOnlineMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, //
      LearningRate learningRate1, LearningRate learningRate2, //
      FeatureWeight w1, FeatureWeight w2) {
    this.monteCarloInterface = monteCarloInterface;
    doubleTrueOnlineSarsa = sarsaType.doubleTrueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate1, learningRate2, w1, w2);
    doubleTrueOnlineSarsa.setExplore(EPSILON);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsa(), EPSILON);
    ExploringStarts.batch(monteCarloInterface, policy, doubleTrueOnlineSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return doubleTrueOnlineSarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepInterface stepInterface) {
    doubleTrueOnlineSarsa.digest(stepInterface);
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return doubleTrueOnlineSarsa.qsaInterface();
  }
}
