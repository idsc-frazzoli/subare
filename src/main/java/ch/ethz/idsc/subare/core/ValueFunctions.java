// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Norm;

/**
 * 
 */
public class ValueFunctions {
  /** under the assumption that p(s',r | s,a) \in {0,1}, i.e.
   * p(s',r | s,a) does not require special treatment or looping...
   * 
   * @param standardModel
   * @param statesIndex
   * @param actionsIndex
   * @param gamma discount
   * @param threshold
   * @return */
  public static Tensor bellmanIteration( //
      StandardModel standardModel, PolicyInterface policyInterface, Index statesIndex, Index actionsIndex, Scalar gamma, Scalar threshold) {
    final int n = statesIndex.size();
    Tensor v = Array.zeros(n); // TODO initial value
    while (true) {
      Tensor v_new = Array.zeros(n); // values are added to 0's during iteration
      for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
        final Tensor state = statesIndex.get(stateI);
        // general bellman equation:
        // v_pi(s) == Sum_a pi(a|s) * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
        // simplifies here to
        // v_pi(s) == Sum_a pi(a|s) * (r + gamma * v_pi(s'))
        for (int actionI = 0; actionI < actionsIndex.size(); ++actionI) {
          final Tensor action = actionsIndex.get(actionI);
          // ---
          Tensor next = standardModel.move(state, action);
          Scalar policy = policyInterface.policy(state, action);
          Scalar reward = standardModel.reward(state, action);
          int nextI = statesIndex.indexOf(next);
          Scalar delta = policy.multiply(reward.add(gamma.multiply(v.Get(nextI))));
          v_new.set(s -> s.add(delta), stateI);
        }
      }
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v)), threshold))
        break;
      v = v_new.unmodifiable();
    }
    return v;
  }
}
