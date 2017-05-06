// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Norm;

/** value iteration: "policy evaluation is stopped after just one sweep"
 * (3.17) on p.69
 * (4.10) on p.89
 * see box on p.90
 * 
 * approximately equivalent to iterating with {@link GreedyPolicy}
 * 
 * parallel implementation
 * initial values are set to zeros
 * Jacobi style, i.e. updates take effect only in the next iteration */
public class ValueIteration {
  final StandardModel standardModel;
  final Scalar gamma;
  private Tensor v_old;
  private Tensor v_new;

  /** @param standardModel
   * @param gamma discount */
  public ValueIteration(StandardModel standardModel, Scalar gamma) {
    this.standardModel = standardModel;
    this.gamma = gamma;
    initialize(Array.zeros(standardModel.states().length()));
  }

  /** @param v_new */
  public void initialize(Tensor v_new) {
    this.v_new = v_new;
  }

  /** perform one step of the iteration
   * 
   * @return */
  public Tensor step() {
    v_old = v_new; // <- preserve old values for advancing iteration via step() and comparison
    Tensor gvalues = v_new.multiply(gamma);
    v_new = Tensor.of(standardModel.states().flatten(0) //
        .parallel() //
        .map(state -> jacobiMax(state, gvalues)));
    return v_new;
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold
   * @return */
  public Tensor untilBelow(Scalar threshold) {
    while (true) {
      step();
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v_old)), threshold))
        return v_new;
    }
  }

  // helper function
  private Scalar jacobiMax(Tensor state, Tensor gvalues) {
    return standardModel.actions(state).flatten(0) //
        .map(action -> standardModel.qsa(state, action, gvalues)) //
        .reduce(Max::of).get();
  }
}
