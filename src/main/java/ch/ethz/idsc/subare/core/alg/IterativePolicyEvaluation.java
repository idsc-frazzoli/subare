// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Norm;

// general bellman equation:
// v_pi(s) == Sum_a pi(a|s) * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
// bellman optimality equation:
// v_*(s) == max_a Sum_{s',r} p(s',r | s,a) * (r + gamma * v_*(s'))
public class IterativePolicyEvaluation {
  StandardModel standardModel;
  PolicyInterface policyInterface;
  private Tensor v_new;
  private int iterations = 0;

  // ---
  /** iterative policy evaluation (4.5)
   * see box on p.81
   * 
   * parallel implementation
   * initial values are set to zeros
   * Jacobi style, i.e. updates take effect only in the next iteration
   * 
   * @param standardModel
   * @param policyInterface
   * @return */
  public IterativePolicyEvaluation( //
      StandardModel standardModel, PolicyInterface policyInterface) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
  }

  /** @param gamma
   * @param threshold
   * @return */
  public Tensor until(Scalar gamma, Scalar threshold) {
    Tensor v_old = Array.zeros(standardModel.states().length());
    while (true) {
      Tensor gvalues = v_old.multiply(gamma);
      v_new = Tensor.of(standardModel.states().flatten(0) //
          .parallel() //
          .map(state -> jacobiAdd(state, gvalues)));
      ++iterations;
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v_old)), threshold))
        break;
      v_old = v_new.unmodifiable();
    }
    return values();
  }

  // helper function
  private Scalar jacobiAdd(Tensor state, Tensor gvalues) {
    return standardModel.actions(state).flatten(0) //
        .map(action -> policyInterface.policy(state, action).multiply(standardModel.qsa(state, action, gvalues))) //
        .reduce(Scalar::add).get();
  }

  public Tensor values() {
    return v_new;
  }

  public int iterations() {
    return iterations;
  }
}
