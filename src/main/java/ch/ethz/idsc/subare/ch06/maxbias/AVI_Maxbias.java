// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.DecimalScalar;

class AVI_Maxbias {
  public static void main(String[] args) {
    Maxbias gridworld = new Maxbias(3);
    ActionValueIteration avi = new ActionValueIteration(gridworld);
    avi.untilBelow(DecimalScalar.of(.0001));
    System.out.println("iterations=" + avi.iterations());
    avi.qsa().print();
    System.out.println("--");
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    dvs.print();
  }
}
