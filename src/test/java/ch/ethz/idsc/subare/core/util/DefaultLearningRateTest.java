// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class DefaultLearningRateTest extends TestCase {
  public void testFirst() {
    LearningRate learningRate = DefaultLearningRate.of(0.9, .51);
    Gambler gambler = new Gambler(100, RealScalar.of(0.4));
    QsaInterface qsa = DiscreteQsa.build(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(gambler, learningRate, qsa, sac, PolicyType.EGREEDY.bestEquiprobable(gambler, qsa, sac));
    Tensor state = Tensors.vector(1);
    Tensor action = Tensors.vector(0);
    Scalar first = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertEquals(first, RealScalar.ONE);
    sarsa.sac().digest(new StepAdapter(state, action, RealScalar.ZERO, state));
    Scalar second = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertTrue(Scalars.lessThan(second, first));
  }

  public void testFailFactor() {
    try {
      DefaultLearningRate.of(0, 1);
      fail();
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(-1, 1);
      fail();
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailExponent() {
    try {
      DefaultLearningRate.of(1, 0.5);
      fail();
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(1, 0.4);
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
