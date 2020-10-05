// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.ch04.gambler.GamblerModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.util.AssertFail;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class DefaultLearningRateTest extends TestCase {
  public void testFirst() {
    LearningRate learningRate = DefaultLearningRate.of(0.9, .51);
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.4));
    QsaInterface qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(gamblerModel, learningRate, qsa, sac, PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac));
    Tensor state = Tensors.vector(1);
    Tensor action = Tensors.vector(0);
    Scalar first = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertEquals(first, RealScalar.ONE);
    sarsa.sac().digest(new StepAdapter(state, action, RealScalar.ZERO, state));
    Scalar second = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertTrue(Scalars.lessThan(second, first));
  }

  public void testFailFactor() {
    AssertFail.of(() -> DefaultLearningRate.of(0, 1));
    AssertFail.of(() -> DefaultLearningRate.of(-1, 1));
  }

  public void testFailExponent() {
    AssertFail.of(() -> DefaultLearningRate.of(1, 0.5));
    AssertFail.of(() -> DefaultLearningRate.of(1, 0.4));
  }
}
