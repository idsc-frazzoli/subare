// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;

public enum MaxbiasHelper {
  ;
  // ---
  static DiscreteQsa exactQsa(Maxbias maxbias) {
    ActionValueIteration avi = new ActionValueIteration(maxbias);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }
}