// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

/* package */ class TrueOnlineMonteCarloTrial implements MonteCarloTrial {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  public static TrueOnlineMonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac = new DiscreteStateActionCounter();
    return new TrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, featureMapper, ConstantLearningRate.of(ALPHA), new FeatureWeight(featureMapper), sac,
        PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final TrueOnlineSarsa trueOnlineSarsa;
  private final PolicyBase policy;

  private TrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, LearningRate learningRate, FeatureWeight w, StateActionCounter sac, PolicyBase policy) {
    this.monteCarloInterface = monteCarloInterface;
    this.policy = policy;
    trueOnlineSarsa = sarsaType.trueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate, w, sac, policy);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    policy.setQsa(trueOnlineSarsa.qsaInterface());
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
