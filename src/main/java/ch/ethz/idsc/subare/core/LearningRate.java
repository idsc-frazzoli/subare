// code by jph
package ch.ethz.idsc.subare.core;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** functionality to implement formula's for learning rate
 * that may depend on the {@link StepInterface} seen in the episodes */
public abstract class LearningRate implements StepDigest {
  /** the map counts the frequency of the state-action pair */
  protected final Map<Tensor, Integer> map = new HashMap<>();

  /** @param stepInterface
   * @return key for identifying steps that are considered identical for counting */
  protected abstract Tensor key(Tensor prev, Tensor action);

  /** successive calls to the function give the same result.
   * 
   * the first call to the function should return numerical value == 1
   * to prevent initialization bias.
   * 
   * the learning rate may chance only upon calling {@link StepDigest#digest}.
   * 
   * @param state
   * @param action
   * @return learning rate for given state-action pair */
  public abstract Scalar alpha(StepInterface stepInterface);

  @Override // from StepDigest
  public synchronized final void digest(StepInterface stepInterface) {
    Tensor key = key(stepInterface.prevState(), stepInterface.action());
    map.put(key, map.containsKey(key) ? map.get(key) + 1 : 1);
  }

  /** function exists to remove the initialization bias
   * 
   * @param state
   * @param action
   * @return whether given (state, action) pair has already been encountered by learning rate */
  public boolean encountered(Tensor state, Tensor action) {
    return map.containsKey(key(state, action));
  }

  /** @param state
   * @param action
   * @return the number of visits occurred with learning rate */
  public int visits(Tensor state, Tensor action) {
    return encountered(state, action) ? map.get(key(state, action)) : 0;
  }
}
