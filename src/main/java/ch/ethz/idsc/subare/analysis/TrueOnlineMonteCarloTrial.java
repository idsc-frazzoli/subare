// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class TrueOnlineMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);
  private static final Scalar EPSILON = RealScalar.of(0.1);

  public static TrueOnlineMonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    return new TrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, featureMapper, ConstantLearningRate.of(ALPHA), new FeatureWeight(featureMapper));
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final TrueOnlineSarsa trueOnlineSarsa;

  private TrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, LearningRate learningRate, FeatureWeight w) {
    this.monteCarloInterface = monteCarloInterface;
    trueOnlineSarsa = sarsaType.trueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate, w);
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

  @Override // from MonteCarloTrial
  public void digest(StepInterface stepInterface) {
    trueOnlineSarsa.digest(stepInterface);
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return trueOnlineSarsa.qsaInterface();
  }
}
