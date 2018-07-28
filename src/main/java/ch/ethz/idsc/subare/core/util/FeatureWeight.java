// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;

public class FeatureWeight implements Serializable {
  /** vector */
  private Tensor w;

  public FeatureWeight(FeatureMapper featureMapper) {
    w = Array.zeros(featureMapper.featureSize());
  }

  public Tensor get() {
    return w;
  }

  public void set(Tensor w) {
    this.w = w;
  }
}
