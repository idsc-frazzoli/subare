// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.DecimalScalar;

enum GridworldHelper {
  ;
  // ---
  static DiscreteQsa exactQsa() {
    Gridworld gridworld = new Gridworld();
    ActionValueIteration avi = new ActionValueIteration(gridworld);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }
}
