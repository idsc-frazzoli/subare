// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.alg.Array;

public class FeatureWeight implements Serializable {
  private final FeatureMapper featureMapper;
  /** vector */
  private Tensor w;

  public FeatureWeight(FeatureMapper featureMapper) {
    this.featureMapper = featureMapper;
    w = Array.zeros(featureMapper.featureSize());
  }

  public Tensor get() {
    return w;
  }

  /** @param w vector of same length as feature size */
  public void set(Tensor w) {
    if (w.length() != featureMapper.featureSize())
      throw TensorRuntimeException.of(w);
    this.w = w;
  }
}
