// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** v_pi(s) := E_pi[ G_t | S_t==s ]
 * 
 * recursive formula
 * v_pi(s) == Sum_a pi(a|s) * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s')) */
public interface ValueFunction {
  Scalar valueOf(Tensor state);
}
