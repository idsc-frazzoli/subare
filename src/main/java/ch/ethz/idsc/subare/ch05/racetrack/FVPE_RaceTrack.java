// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.sca.Round;

class FVPE_RaceTrack {
  public static void main(String[] args) throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        racetrack, RealScalar.ONE, null);
    PolicyInterface policyInterface = new EquiprobablePolicy(racetrack);
    for (int count = 0; count < 10; ++count)
      ExploringStartsBatch.apply(racetrack, fvpe, policyInterface);
    DiscreteVs vs = fvpe.vs();
    vs.print(Round.toMultipleOf(DecimalScalar.of(.1)));
  }
}
