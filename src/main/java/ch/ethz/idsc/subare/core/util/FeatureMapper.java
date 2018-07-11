// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Tensor;

public interface FeatureMapper {
  /** @param key for instance {@link StateAction#key(Tensor, Tensor)}, or {@link StateAction#key(StepInterface)}
   * @return TODO document is return value always a vector? ... that should combined with a dot product? */
  Tensor getFeature(Tensor key);

  /** @return TODO document */
  int stateActionSize(); // TODO function is not used yet

  /** @return TODO document */
  int featureSize();
}
