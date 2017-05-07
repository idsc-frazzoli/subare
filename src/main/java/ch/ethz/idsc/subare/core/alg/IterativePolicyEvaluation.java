// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
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
  private final StandardModel standardModel;
  private final PolicyInterface policyInterface;
  private final Scalar gamma;
  private Tensor v_old;
  private Tensor v_new;
  VsInterface vs_new;
  VsInterface vs_old;
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
      StandardModel standardModel, PolicyInterface policyInterface, Scalar gamma) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    this.gamma = gamma;
    v_new = Array.zeros(standardModel.states().length());
    vs_new = DiscreteVs.build(standardModel);
    vs_old = DiscreteVs.build(standardModel);
  }

  /** @param gamma
   * @param threshold
   * @return */
  public Tensor until(Scalar threshold) {
    Scalar past = null;
    while (true) {
      step();
      Scalar delta = Norm._1.of(v_new.subtract(v_old));
      if (past != null && Scalars.lessThan(past, delta)) {
        System.out.println("give up at " + past + " -> " + delta);
        return v_old;
      }
      past = delta;
      if (Scalars.lessThan(delta, threshold))
        break;
    }
    return values();
  }

  public Tensor step() {
    v_old = v_new; // <- preserve old values for advancing iteration via step() and comparison
    Tensor gvalues = v_old.multiply(gamma);
    v_new = Tensor.of(standardModel.states().flatten(0) //
        .parallel() //
        .map(state -> jacobiAdd(state, gvalues)));
    ++iterations;
    return v_new;
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
