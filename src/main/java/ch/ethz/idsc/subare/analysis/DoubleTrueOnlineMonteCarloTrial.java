// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.Objects;

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
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleTrueOnlineSarsa doubleTrueOnlineSarsa;

  // has convergence problems, don't use it yet!
  public DoubleTrueOnlineMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      LearningRate learningRate1_, LearningRate learningRate2_, //
      FeatureWeight w1_, FeatureWeight w2_) {
    this.monteCarloInterface = monteCarloInterface;
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    LearningRate learningRate1 = Objects.isNull(learningRate1_) ? ConstantLearningRate.of(ALPHA) : learningRate1_;
    LearningRate learningRate2 = Objects.isNull(learningRate2_) ? ConstantLearningRate.of(ALPHA) : learningRate2_;
    FeatureWeight w1 = Objects.isNull(w1_) ? new FeatureWeight(featureMapper) : w1_;
    FeatureWeight w2 = Objects.isNull(w2_) ? new FeatureWeight(featureMapper) : w2_;
    doubleTrueOnlineSarsa = sarsaType.doubleTrueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate1, learningRate2, w1, w2);
    doubleTrueOnlineSarsa.setExplore(EPSILON);
  }

  public DoubleTrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    this(monteCarloInterface, sarsaType, null, null, null, null);
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
