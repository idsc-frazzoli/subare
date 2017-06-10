// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** single action sarsa */
abstract class ActionSarsa extends Sarsa {
  public ActionSarsa(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    super(discreteModel, qsa, alpha);
  }

  /** the action "has the value" estimated by evaluate(...)
   * 
   * @param state
   * @return action from state using policy derived from Q */
  abstract Tensor selectAction(Tensor state);

  @Override
  protected final Scalar evaluate(Tensor state) {
    return qsa.value(state, selectAction(state));
  }
}
