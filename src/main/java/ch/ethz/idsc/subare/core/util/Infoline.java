// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Round;

public enum Infoline {
  ;
  public static void print(DiscreteModel discreteModel, int index, DiscreteQsa ref, DiscreteQsa qsa) {
    Scalar error = DiscreteValueFunctions.distance(qsa, ref);
    Scalar loss = Loss.accumulation(discreteModel, DiscreteValueFunctions.numeric(ref), qsa);
    System.out.println(String.format("%2d %8s  ", index, error.map(Round._1)) + loss);
  }
}
