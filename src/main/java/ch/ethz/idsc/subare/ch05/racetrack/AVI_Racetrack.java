// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Import;

/** action value iteration for gambler's dilemma */
class AVI_Racetrack {
  public static final String FILENAME = "racetrack_avi_qsa.object";

  static void precompute() throws Exception {
    String trackName = "track2";
    Racetrack racetrack = RacetrackHelper.create(trackName, 5);
    ActionValueIteration avi = new ActionValueIteration(racetrack, racetrack);
    avi.untilBelow(RealScalar.of(1e-1), 0);
    Export.object(UserHome.file(FILENAME), avi.qsa());
  }

  public static void main(String[] args) throws Exception {
    // precompute();
    DiscreteQsa qsa = Import.object(UserHome.file(FILENAME));
    System.out.println(qsa.size());
    String trackName = "track2";
    Racetrack racetrack = RacetrackHelper.create(trackName, 5);
    int c = 0;
    for (Tensor state : racetrack.states()) {
      for (Tensor action : racetrack.actions(state)) {
        try {
          qsa.value(state, action);
          ++c;
        } catch (Exception exception) {
          // ---
        }
      }
    }
    System.out.println(c + " / " + qsa.size());
    Export.of(UserHome.file("Pictures/racetrack_qsa_avi_21_11.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(1, 1)));
    Export.of(UserHome.file("Pictures/racetrack_qsa_avi_21_10.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(1, 0)));
    Export.of(UserHome.file("Pictures/racetrack_qsa_avi_21_01.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(0, 1)));
  }
}
