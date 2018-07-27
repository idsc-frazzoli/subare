// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import junit.framework.TestCase;

public class TrueOnlineSarsaTest extends TestCase {
  public void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
      FeatureWeight w = new FeatureWeight(featureMapper);
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline( //
          SimpleTestModel.INSTANCE, RealScalar.ONE, featureMapper, learningRate, w);
      Scalar epsilon = RealScalar.of(.2);
      trueOnlineSarsa.setExplore(epsilon);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, trueOnlineSarsa.qsa(), epsilon);
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
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline( //
          SimpleTestModel.INSTANCE, RealScalar.of(.9), featureMapper, learningRate, w);
      Scalar epsilon = RealScalar.of(.2);
      trueOnlineSarsa.setExplore(epsilon);
      for (int index = 0; index < 10; ++index) {
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, trueOnlineSarsa.qsa(), epsilon);
        ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
      }
      DiscreteQsa qsa = trueOnlineSarsa.qsa();
      DiscreteUtils.print(qsa);
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
      SarsaType.ORIGINAL.trueOnline(monteCarloInterface, RealScalar.of(2), featureMapper, learningRate, w);
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
      SarsaType.ORIGINAL.trueOnline(null, RealScalar.ONE, featureMapper, learningRate, w);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailEpsilon() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = new Gambler(10, RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    TrueOnlineSarsa trueOnlineSarsa = SarsaType.ORIGINAL.trueOnline(monteCarloInterface, RealScalar.ONE, featureMapper, learningRate, w);
    try {
      trueOnlineSarsa.setExplore(RealScalar.of(-0.1));
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
