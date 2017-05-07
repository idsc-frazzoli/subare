// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
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
  private final StandardModel standardModel;
  private final Scalar gamma;
  private final DiscreteVs vs_new;
  private final DiscreteVs vs_old;
  int iterations = 0;
  int alternate = 0;

  /** @param standardModel
   * @param gamma discount */
  public ValueIteration(StandardModel standardModel, Scalar gamma) {
    this.standardModel = standardModel;
    this.gamma = gamma;
    vs_new = DiscreteVs.build(standardModel);
    vs_old = DiscreteVs.build(standardModel);
  }

  /** perform one step of the iteration
   * 
   * @return */
  public void step() {
    vs_old.setAll(vs_new.values());
    Tensor gvalues = vs_new.values().multiply(gamma);
    vs_new.setAll(Tensor.of(standardModel.states().flatten(0) //
        .parallel() //
        .map(state -> jacobiMax(state, gvalues))));
    ++iterations;
  }

  /** perform iteration until values don't change more than threshold
   * 
   * @param threshold
   * @return */
  public void untilBelow(Scalar threshold) {
    untilBelow(threshold, Integer.MAX_VALUE);
  }

  public void untilBelow(Scalar threshold, int flips) {
    Scalar past = null;
    while (true) {
      step();
      Scalar delta = Norm._1.of(vs_new.values().subtract(vs_old.values()));
      if (past != null && Scalars.lessThan(past, delta)) {
        if (flips < ++alternate) {
          System.out.println("give up at " + past + " -> " + delta);
          break;
        }
      }
      past = delta;
      if (Scalars.lessThan(delta, threshold))
        break;
    }
  }

  // helper function
  private Scalar jacobiMax(Tensor state, Tensor gvalues) {
    return standardModel.actions(state).flatten(0) //
        .map(action -> standardModel.qsa(state, action, gvalues)) //
        .reduce(Max::of).get();
  }

  public DiscreteVs vs() {
    return vs_new;
  }

  public int iterations() {
    return iterations;
  }
}
