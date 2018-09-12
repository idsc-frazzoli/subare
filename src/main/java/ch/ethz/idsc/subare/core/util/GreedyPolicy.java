// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.RealScalar;

public enum GreedyPolicy {
  ;
  public static Policy of(DiscreteModel discreteModel, QsaInterface qsa) {
    return new EGreedyPolicy(discreteModel, qsa, RealScalar.ZERO);
  }

  public static Policy of(StandardModel standardModel, VsInterface vs) {
    return new EGreedyPolicy(standardModel, vs, RealScalar.ZERO, standardModel.states());
  }
}
