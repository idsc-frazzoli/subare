// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** Q-learning: An off-policy TD control algorithm
 * 
 * eq (6.8)
 * 
 * box on p.140
 * 
 * see also Watkins 1989 */
// TODO check convergence criteria
// learning rate should converge to zero, sum should go to infinity, sum of squares should be finite(?)
public class QLearning extends Sarsa {
  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate */
  public QLearning(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    super(discreteModel, qsa, alpha);
  }

  @Override
  protected Scalar evaluate(Tensor state1) {
    return discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
  }
}
