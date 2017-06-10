// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StandardModel extends DiscreteModel {
  /** function implements the formula
   * Sum_{s',r} p(s',r | s,a) * [r + gamma * v_*(s')]
   * 
   * general term in bellman equation:
   * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
   * 
   * where
   * v_*(s) == max_a q_*(s, a)
   * 
   * @param state
   * @param action
   * @param gvalues value function already discounted by gamma
   * @return expected return for the best action for that state */
  Scalar qsa(Tensor state, Tensor action, VsInterface gvalues);
}
