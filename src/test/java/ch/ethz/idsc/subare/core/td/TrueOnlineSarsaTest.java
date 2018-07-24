// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import junit.framework.TestCase;

public class TrueOnlineSarsaTest extends TestCase {
  public void testFailLambda() {
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    Tensor w = Array.zeros(featureMapper.featureSize());
    try {
      SarsaType.ORIGINAL.trueOnline(monteCarloInterface, RealScalar.of(2), featureMapper, learningRate, w);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFail() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    Tensor w = Array.zeros(featureMapper.featureSize());
    try {
      SarsaType.ORIGINAL.trueOnline(null, RealScalar.ONE, featureMapper, learningRate, w);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailEpsilon() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    Tensor w = Array.zeros(featureMapper.featureSize());
    TrueOnlineSarsa trueOnlineSarsa = SarsaType.ORIGINAL.trueOnline(monteCarloInterface, RealScalar.ONE, featureMapper, learningRate, w);
    try {
      trueOnlineSarsa.setExplore(RealScalar.of(-0.1));
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
