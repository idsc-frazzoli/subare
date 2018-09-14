// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
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

public class DoubleTrueOnlineMonteCarloTrial implements MonteCarloTrial, StateActionCounterSupplier {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  public static DoubleTrueOnlineMonteCarloTrial create(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
    PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    return new DoubleTrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, featureMapper, //
        ConstantLearningRate.of(ALPHA), sac1, sac2, //
        new FeatureWeight(featureMapper), new FeatureWeight(featureMapper), //
        policy1, policy2);
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleTrueOnlineSarsa doubleTrueOnlineSarsa;

  // has convergence problems, don't use it yet!
  private DoubleTrueOnlineMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      StateActionCounter sac1, StateActionCounter sac2, //
      FeatureWeight w1, FeatureWeight w2, //
      PolicyBase policy1, PolicyBase policy2) {
    this.monteCarloInterface = monteCarloInterface;
    doubleTrueOnlineSarsa = sarsaType.doubleTrueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate, w1, w2, sac1, sac2, policy1, policy2);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, doubleTrueOnlineSarsa.getPolicy(), doubleTrueOnlineSarsa);
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

  @Override
  public StateActionCounter sac() {
    return doubleTrueOnlineSarsa.sac();
  }
}
