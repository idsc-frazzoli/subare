package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Join;

public class ExactMapper extends FeatureMapper {
  public ExactMapper(MonteCarloInterface mcInterface) {
    super(mcInterface);
  }

  @Override
  protected void initMap(MonteCarloInterface mcInterface) {
    // count the number of possible state-action pairs first
    int count = 0;
    for (Tensor state : mcInterface.states()) {
      count += mcInterface.actions(state).length();
    }
    stateActionSize = count;
    featureSize = count; // one-to-one mapping
    int index = 0;
    for (Tensor state : mcInterface.states()) {
      for (Tensor action : mcInterface.actions(state)) {
        final int i = index;
        Tensor feature = Tensors.vector(v -> (v == i ? RealScalar.ONE : RealScalar.ZERO), stateActionSize);
        stateToFeature.put(Join.of(state, action), feature);
        ++index;
      }
    }
  }
}
