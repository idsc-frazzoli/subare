// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.Tensor;

public interface FeatureMapper {
  /** Example of key is Join.of(state, action)
   * 
   * @param key
   * @return */
  Tensor getFeature(Tensor key);

  int getStateActionSize();

  int getFeatureSize();
}
