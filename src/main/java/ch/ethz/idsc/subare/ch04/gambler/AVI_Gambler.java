// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.Settings;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Put;

/**  */
class AVI_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler, RealScalar.ONE);
    avi.untilBelow(RealScalar.of(1e-3));
    avi.qsa();
    DiscreteVs dvs = DiscreteUtils.createVs(gambler, avi.qsa());
    dvs.print();
    Put.of(new File(Settings.home(), "ex403_qsa_values"), dvs.values());
  }
}
