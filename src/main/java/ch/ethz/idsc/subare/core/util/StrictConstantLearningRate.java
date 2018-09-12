// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;

/** THE USE OF THIS CLASS IS NOT RECOMMENDED BECAUSE THE
 * UPDATE IS BIASED TOWARDS AN UNWARRANTED INITIAL VALUE */
public class StrictConstantLearningRate implements LearningRate, Serializable {
  private final Scalar alpha;

  public StrictConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface, StateActionCounter stateActionCounter) {
    return alpha;
  }
}
