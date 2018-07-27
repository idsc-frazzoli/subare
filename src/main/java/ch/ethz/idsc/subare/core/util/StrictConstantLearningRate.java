// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** THE USE OF THIS CLASS IS NOT RECOMMENDED BECAUSE THE
 * UPDATE IS BIASED TOWARDS AN UNWARRANTED INITIAL VALUE */
public class StrictConstantLearningRate implements LearningRate, Serializable {
  private final Scalar alpha;

  public StrictConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface) {
    return alpha;
  }

  @Override
  public void digest(StepInterface stepInterface) {
    // ---
  }

  @Override
  public boolean isEncountered(Tensor state, Tensor action) {
    return false; // not defined
  }
}
