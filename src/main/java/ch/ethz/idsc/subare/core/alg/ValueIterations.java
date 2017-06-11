// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.Scalar;

public enum ValueIterations {
  ;
  // ---
  public static DiscreteVs solve(StandardModel standardModel, Scalar threshold) {
    ValueIteration valueIteration = new ValueIteration(standardModel);
    valueIteration.untilBelow(threshold);
    return valueIteration.vs();
  }
}
