// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashSet;
import java.util.Set;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1. */
public class ConstantLearningRate implements LearningRate {
  public static LearningRate of(Scalar alpha) {
    return new ConstantLearningRate(alpha);
  }

  // ---
  private final Set<Tensor> visited = new HashSet<>();
  private final Scalar alpha;

  private ConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override
  public void digest(StepInterface stepInterface) {
    visited.add(StateAction.key(stepInterface));
  }

  @Override
  public Scalar alpha(StepInterface stepInterface) {
    boolean visited1 = visited.contains(StateAction.key(stepInterface));
    return visited1 ? alpha : RealScalar.ONE; // overcome initialization bias
  }
}
