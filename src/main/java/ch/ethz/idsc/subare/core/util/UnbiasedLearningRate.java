// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteQsaWeight;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Tensor;

abstract class UnbiasedLearningRate implements LearningRate, DiscreteQsaWeight, Serializable {
  /** the map counts the frequency of the state-action pair */
  private final Map<Tensor, Integer> map = new HashMap<>();

  @Override // from StepDigest
  public synchronized final void digest(StepInterface stepInterface) {
    Tensor key = key(stepInterface.prevState(), stepInterface.action());
    map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
  }

  @Override // from LearningRate
  public final boolean isEncountered(Tensor state, Tensor action) {
    Tensor key = key(state, action);
    return map.containsKey(key);
  }

  @Override // from DiscreteQsaWeight
  public final int counts(Tensor key) {
    return map.containsKey(key) ? map.get(key) : 0;
  }

  /** @param stepInterface
   * @return key for identifying steps that are considered identical for counting */
  protected abstract Tensor key(Tensor prev, Tensor action);
}
