// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Norm;

// general bellman equation:
// v_pi(s) == Sum_a pi(a|s) * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
// bellman optimality equation:
// v_*(s) == max_a Sum_{s',r} p(s',r | s,a) * (r + gamma * v_*(s'))
public class ValueFunctions {
  /** iterative policy evaluation (4.5)
   * see box on p.81
   * 
   * parallel implementation
   * initial values are set to zeros
   * Jacobi style, i.e. updates take effect only in the next iteration
   * 
   * @param standardModel
   * @param policyInterface
   * @param gamma discount
   * @param threshold
   * @return */
  public static Tensor bellmanIteration( //
      StandardModel standardModel, PolicyInterface policyInterface, Scalar gamma, Scalar threshold) {
    Tensor v_old = Array.zeros(standardModel.states().length());
    while (true) {
      Tensor gvalues = v_old.multiply(gamma);
      Tensor v_new = Tensor.of(standardModel.states().flatten(0) //
          .parallel() //
          .map(state -> jacobiAdd(standardModel, policyInterface, state, gvalues)));
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v_old)), threshold))
        return v_new;
      v_old = v_new.unmodifiable();
    }
  }

  // helper function
  private static Scalar jacobiAdd(StandardModel standardModel, PolicyInterface policyInterface, Tensor state, Tensor gvalues) {
    return standardModel.actions(state).flatten(0) //
        .map(action -> policyInterface.policy(state, action).multiply(standardModel.qsa(state, action, gvalues))) //
        .reduce(Scalar::add).get();
  }

}
