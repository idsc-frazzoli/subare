// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class DefaultLearningRateTest extends TestCase {
  public void testFirst() {
    LearningRate learningRate = DefaultLearningRate.of(0.9, .51);
    Tensor state = Tensors.vector(1);
    Tensor action = Tensors.vector(0);
    Scalar first = learningRate.alpha(state, action);
    assertEquals(first, RealScalar.ONE);
    learningRate.digest(new StepAdapter(state, action, RealScalar.ZERO, state));
    Scalar second = learningRate.alpha(state, action);
    // System.out.println(second);
    assertTrue(Scalars.lessThan(second, first));
  }

  public void testFailFactor() {
    try {
      DefaultLearningRate.of(0, 1);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(-1, 1);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailExponent() {
    try {
      DefaultLearningRate.of(1, 0.5);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(1, 0.4);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
