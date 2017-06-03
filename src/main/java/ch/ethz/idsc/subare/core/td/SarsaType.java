// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;

public enum SarsaType {
  original, //
  expected, //
  ;
  // ---
  public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha, //
      PolicyInterface policyInterface) {
    switch (this) {
    case original:
      return new OriginalSarsa(discreteModel, qsa, alpha, policyInterface);
    case expected:
      return new ExpectedSarsa(discreteModel, qsa, alpha, policyInterface);
    }
    throw new RuntimeException();
  }
}
