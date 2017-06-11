// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.DecimalScalar;

class AVI_Maxbias {
  public static void main(String[] args) {
    Maxbias gridworld = new Maxbias(3);
    DiscreteQsa qsa = ActionValueIterations.solve(gridworld, DecimalScalar.of(.0001));
    qsa.print();
    System.out.println("---");
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, qsa);
    dvs.print();
  }
}
