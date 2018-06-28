// code by fluric
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.UnitVector;

/** requires keys of the form Join.of(state, action)
 * 
 * the implementation initializes the features as unit vectors */
public class ExactFeatureMapper implements FeatureMapper {
  private final Map<Tensor, Tensor> stateToFeature = new HashMap<>();
  private final int stateActionSize;
  private final int featureSize;

  public ExactFeatureMapper(MonteCarloInterface mcInterface) {
    // count the number of possible state-action pairs first
    int count = 0;
    for (Tensor state : mcInterface.states()) {
      count += mcInterface.actions(state).length();
    }
    stateActionSize = count;
    featureSize = count; // one-to-one mapping
    int index = -1;
    for (Tensor state : mcInterface.states())
      for (Tensor action : mcInterface.actions(state))
        stateToFeature.put(Join.of(state, action), UnitVector.of(stateActionSize, ++index));
  }

  @Override // from FeatureMapper
  public Tensor getFeature(Tensor key) {
    return stateToFeature.get(key);
  }

  @Override // from FeatureMapper
  public int getStateActionSize() {
    return stateActionSize;
  }

  @Override // from FeatureMapper
  public int getFeatureSize() {
    return featureSize;
  }
}
