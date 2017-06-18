// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.sca.Round;

class FVPE_RaceTrack {
  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        racetrack, null);
    Policy policy = new EquiprobablePolicy(racetrack);
    for (int count = 0; count < 10; ++count)
      ExploringStarts.batch(racetrack, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    vs.print(Round._1);
  }
}
