package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.tensor.Tensor;

public abstract class FeatureMapper {
  protected Map<Tensor, Tensor> stateToFeature = new HashMap<>();
  protected int stateActionSize;
  protected int featureSize;

  public FeatureMapper(MonteCarloInterface mcInterface) {
    initMap(mcInterface);
  }

  protected abstract void initMap(MonteCarloInterface mcInterface);

  public Tensor getFeature(Tensor state) {
    return stateToFeature.get(state);
  }

  public int getStateActionSize() {
    return stateActionSize;
  }

  public int getFeatureSize() {
    return featureSize;
  }
}
