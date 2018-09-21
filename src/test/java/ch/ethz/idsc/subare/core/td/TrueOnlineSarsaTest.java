// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class TrueOnlineSarsaTest extends TestCase {
  public void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
      FeatureWeight w = new FeatureWeight(featureMapper);
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(SimpleTestModel.INSTANCE, RealScalar.ONE, featureMapper, //
          learningRate, w, sac, policy);
      ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
      // DiscreteUtils.print(trueOnlineSarsa.qsa());
      SimpleTestModels._checkExact(trueOnlineSarsa.qsa());
    }
  }

  public void testLambda() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
      FeatureWeight w = new FeatureWeight(featureMapper);
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(SimpleTestModel.INSTANCE, RealScalar.of(0.9), featureMapper, //
          learningRate, w, sac, policy);
      for (int index = 0; index < 10; ++index) {
        ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
      }
      DiscreteUtils.print(trueOnlineSarsa.qsa());
      // TODO doesn't work
      // SimpleTestModels._checkClose(qsa);
    }
  }

  public void testFailLambda() {
    MonteCarloInterface monteCarloInterface = new Gambler(10, RationalScalar.HALF);
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    try {
      SarsaType.ORIGINAL.trueOnline(SimpleTestModel.INSTANCE, RealScalar.of(2), featureMapper, //
          learningRate, w, new DiscreteStateActionCounter(), null);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFail() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = new Gambler(10, RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    try {
      SarsaType.ORIGINAL.trueOnline(null, RealScalar.of(0.9), featureMapper, //
          learningRate, w, new DiscreteStateActionCounter(), null);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
