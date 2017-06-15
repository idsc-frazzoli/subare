// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.sca.Round;

class IPE_RaceTrack {
  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    PolicyInterface policyInterface = new EquiprobablePolicy(racetrack);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation(racetrack, policyInterface);
    ipe.until(RealScalar.of(.1));
    ipe.vs().print(Round._1);
  }
}
