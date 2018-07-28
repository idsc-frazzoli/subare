// code by jph and fluric
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Tensor;

/** interface implemented by classes that also implement {@link LearningRate}
 * to indicate how often the qsa value has been updated due to learning */
public interface DiscreteQsaWeight {
  /** @param key same as listed in {@link DiscreteQsa#keys()}
   * @return number of updates of qsa value for given state-action pair due to learning */
  int counts(Tensor key);
}
