// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Scalar;

/**  */
public enum ActionValueIterations {
  ;
  // ---
  public static DiscreteQsa solve(StandardModel standardModel, Scalar threshold) {
    ActionValueIteration actionValueIteration = new ActionValueIteration(standardModel);
    actionValueIteration.untilBelow(threshold);
    return actionValueIteration.qsa();
  }
}
