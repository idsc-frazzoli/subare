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
import junit.framework.TestCase;

public class TrueOnlineSarsaTest extends TestCase {
  public void testFailLambda() {
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    try {
      TrueOnlineSarsa.of(monteCarloInterface, RealScalar.of(2), learningRate, featureMapper);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFail() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    try {
      TrueOnlineSarsa.of(null, RealScalar.ONE, learningRate, featureMapper);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailEpsilon() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    TrueOnlineSarsa trueOnlineSarsa = TrueOnlineSarsa.of(monteCarloInterface, RealScalar.ONE, learningRate, featureMapper);
    Tensor startState = monteCarloInterface.startStates().get(0);
    Tensor startAction = monteCarloInterface.actions(startState).get(0);
    try {
      trueOnlineSarsa.executeEpisode(RealScalar.of(-0.1), startState, startAction);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}