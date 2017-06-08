// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

abstract class ActionSarsa extends Sarsa {
  public ActionSarsa(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    super(discreteModel, qsa, alpha);
  }

  /** TODO the action "has the value" estimated by evaluate(...)
   * 
   * @param state
   * @return action from state using policy derived from Q */
  protected abstract Tensor chooseAction(Tensor state);

  @Override
  protected final Scalar evaluate(Tensor state) {
    // TODO remove commented old code from qlearning
    // return discreteModel.actions(state).flatten(0) //
    // .map(action1 -> qsa.value(state, action1)) //
    // .reduce(Max::of).get();
    return qsa.value(state, chooseAction(state));
  }
}
